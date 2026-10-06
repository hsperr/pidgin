"""Why the net bid that: sampled hidden hands, weighted by the net itself, played out.

Three pieces, all from the deployed policy alone (no belief net, no double dummy):

- ``sample_owners`` deals the 39 unseen cards at random into the other three seats;
- ``auction_weights`` scores every sample by how likely *this net* thinks the calls
  already made were, holding those cards: log w = sum log p(call | sampled hand, prefix)
  over the other seats' calls. Normalised, that is the net's own picture of the table.
- ``continuations`` appends a candidate call and lets the net finish the auction on every
  sample, then groups identical auctions and sorts them by weight.

The samples are a guess, not the truth: measured on held-out deals, the real hand ranks
around the 18th percentile of the sampled ones. Label them as the net's imagination.
"""
import numpy as np
import torch

from bridgezero.bridge.auction import AuctionState
from emergent.deck import NAMES, call_name

MAX_CONTINUATION = 16          # safety stop for the played-out auction


def sample_owners(bitmaps, actor_seat, n, rng):
    """(n, 4, 52) float hands: the actor's real cards, the rest dealt at random 13/13/13."""
    mine = np.asarray(bitmaps[actor_seat], dtype=np.float32)
    unseen = np.flatnonzero(mine == 0)
    others = [(actor_seat + k) % 4 for k in (1, 2, 3)]
    hands = np.zeros((n, 4, 52), dtype=np.float32)
    hands[:, actor_seat] = mine
    for i in range(n):
        shuffled = rng.permutation(unseen)
        for k, seat in enumerate(others):
            hands[i, seat, shuffled[13 * k:13 * (k + 1)]] = 1.0
    return hands


@torch.no_grad()
def _log_probs(bot, hands, history, seat, legal_mask, vul=(False, False)):
    """(B, n_actions) log policy for one seat holding each sampled hand at one prefix."""
    hist = torch.as_tensor(history, dtype=torch.long)[None].expand(len(hands), -1)
    feats = bot.features(hist, torch.full((len(hands),), seat, dtype=torch.long), vul)
    out = bot.net(torch.as_tensor(hands), feats)
    mask = torch.as_tensor(legal_mask[:bot.net.n_actions], dtype=torch.bool)[None]
    return bot.log_probs(out, mask.expand(len(hands), -1))


def auction_weights(bot, calls, hands, actor_seat, eps=0.02, target=0.25, vul=(False, False)):
    """Weights over the samples, plus the effective sample size and the softening used.

    A confident net gives one lucky sample almost all the weight, which makes the ranking
    meaningless. Two guards, both standard importance-sampling practice: mix ``eps`` of a
    flat policy into every call, and flatten the log weights by ``beta`` chosen so the
    effective sample size reaches ``target`` of the samples.
    """
    n = len(hands)
    log_w = torch.zeros(n)
    st = AuctionState()
    for t, call in enumerate(calls):
        seat = t % 4
        if seat != actor_seat:                       # the actor's own calls carry no information
            mask = bot.mask(st)
            lp = _log_probs(bot, hands[:, seat], calls[:t], seat, mask, vul)
            flat = float(np.log(eps / max(1, int(mask[:bot.net.n_actions].sum()))))
            log_w += torch.logaddexp(lp[:, call] + float(np.log(1 - eps)),
                                     torch.full((n,), flat))
        st = st.apply(call)

    def ess_of(beta):
        w = torch.softmax(beta * log_w, 0)
        return w, float(1.0 / (w * w).sum())

    beta, (w, ess) = 1.0, ess_of(1.0)
    if ess < target * n:                             # flatten until enough samples count
        lo, hi = 0.0, 1.0
        for _ in range(20):
            beta = 0.5 * (lo + hi)
            w, ess = ess_of(beta)
            if ess < target * n:
                hi = beta
            else:
                lo = beta
        w, ess = ess_of(lo if lo > 0 else 1e-3)
        beta = lo
    return w, ess, round(beta, 3)


@torch.no_grad()
def _batch_step(bot, states, hands, live, vul=(False, False)):
    """Greedy call for every live sample, one forward pass over the padded histories."""
    rows = [i for i in live]
    width = max(len(states[i].calls) for i in rows)
    hist = torch.full((len(rows), width), -1, dtype=torch.long)
    seats = torch.empty(len(rows), dtype=torch.long)
    hand = torch.empty(len(rows), 52)
    legal = torch.zeros(len(rows), bot.net.n_actions, dtype=torch.bool)
    for j, i in enumerate(rows):
        st = states[i]
        hist[j, :len(st.calls)] = torch.as_tensor(st.calls, dtype=torch.long)
        seats[j] = st.turn
        hand[j] = torch.as_tensor(hands[i, st.turn])
        legal[j] = torch.as_tensor(bot.mask(st)[:bot.net.n_actions], dtype=torch.bool)
    feats = bot.features(hist, seats, vul)
    out = bot.net(hand, feats)
    return rows, bot.log_probs(out, legal).argmax(-1).tolist()


def continuations(bot, calls, hands, weights, first_call, top=5, vul=(False, False)):
    """Play the auction out after ``first_call`` on every sample; group and rank the results."""
    n = len(hands)
    states = []
    for _ in range(n):
        st = AuctionState()
        for c in calls:
            st = st.apply(c)
        states.append(st.apply(first_call))
    for _ in range(MAX_CONTINUATION):
        live = [i for i in range(n) if not states[i].ended]
        if not live:
            break
        rows, picks = _batch_step(bot, states, hands, live, vul)
        for i, call in zip(rows, picks):
            states[i] = states[i].apply(int(call))
    groups = {}
    for i, st in enumerate(states):
        tail = tuple(st.calls[len(calls):])
        g = groups.setdefault(tail, {"weight": 0.0, "state": st})
        g["weight"] += float(weights[i])
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["weight"])[:top]
    out = []
    for tail, g in ranked:
        st = g["state"]
        contract = None if st.last_contract < 0 else NAMES[st.last_contract]
        if contract and st.doubled:
            contract += "X" * st.doubled
        out.append({"calls": [call_name(c) for c in tail], "share": round(g["weight"], 3),
                    "contract": contract, "declarer": None if contract is None else st.declarer(),
                    "ended": bool(st.ended)})
    return out


@torch.no_grad()
def explain(bot, game, candidates, samples=256, seed=0, top=5):
    """The whole job for one position: sample, weigh, then play out each candidate call.

    North deals; `game["vul"]` (N/S, E/W) defaults to nobody. The table turns a board with
    another dealer round before it gets here."""
    calls = list(game["calls"])
    vul = game.get("vul", (False, False))
    actor = len(calls) % 4
    rng = np.random.default_rng(seed)
    hands = sample_owners(game["bitmaps"], actor, samples, rng)
    weights, ess, beta = auction_weights(bot, calls, hands, actor, vul=vul)
    seen = torch.as_tensor(hands)[:, [(actor + k) % 4 for k in (1, 2, 3)]]
    hcp = (seen.view(samples, 3, 4, 13)[:, :, :, :4] * torch.tensor([4., 3., 2., 1.])).sum((2, 3))
    picture = [{"seat": (actor + k + 1) % 4,
                "hcp_mean": round(float((hcp[:, k] * weights).sum()), 1),
                "hcp_range": [round(float(x), 1) for x in _weighted_range(hcp[:, k], weights)]}
               for k in range(3)]
    return {"samples": samples, "ess": round(ess, 1), "beta": beta, "picture": picture,
            "candidates": [{"call": call_name(c),
                            "lines": continuations(bot, calls, hands, weights, c, top, vul)}
                           for c in candidates]}


def _weighted_range(values, weights, lo=0.1, hi=0.9):
    """Weighted 10th and 90th percentile of one column."""
    order = torch.argsort(values)
    v, w = values[order], weights[order]
    cum = torch.cumsum(w, 0)
    return [float(v[int(torch.searchsorted(cum, torch.tensor(q)).clamp(max=len(v) - 1))])
            for q in (lo, hi)]
