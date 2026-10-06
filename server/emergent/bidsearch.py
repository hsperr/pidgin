"""Bidding search for /debug: sample the hidden hands from a belief net, play the
auction out with the bidding net, score the contracts double dummy.

A port of bridge_new/experiments/belief_multi (bid_search.py, sample_eval.shape; E57
in its NOTES.md). For the seat on turn:

1. Candidates: the bidding net's greedy call plus its next best calls, up to K in all,
   each with p >= PMIN. One candidate: no search.
2. N deals consistent with the hand and the auction, drawn shape-first from the belief
   net (runs/r2: an MLP with per-hand HCP and suit-length heads; system tags "not told"
   for both sides): suit lengths from the length heads, honours by the per-card owner
   probabilities, small cards at random, then resampled by the HCP head.
3. For every deal x candidate: append the candidate, then the acting bidding net bids
   greedily at all four seats until the auction ends.
4. Score each final contract on that deal's double dummy result (doubles included), from
   the acting side's view. Pick the best mean; leave the greedy call only if the best
   beats it by more than MARGIN points.

Research numbers (N=32, K=3, margin 50, shape sampler, r2): +0.37 IMP/board paired gain
for g_s2o_hi3_lo with search vs brl, +0.20 for D75 vs D75.

Differences from the research code, none of which change a result:
- Only the contracts the rollouts reach are solved (DDS `analyse_start`), not the whole
  20-entry table; the scores are the same table entries.
- The belief net's weights are stored in float16 (half the memory; the droplet has
  450 MB for everything) and run in float32, the first layer as a sparse gather.
- A wall-clock budget: deals are solved one at a time, and when the budget runs out the
  search decides on the deals solved so far, or falls back to the greedy call if fewer
  than MIN_SAMPLES are done.
"""
from __future__ import annotations

import math
import sys
import threading
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from training.bridge.auction import AuctionState
from training.bridge.calls import CONTRACTS
from training.bridge.scoring import contract_score
from emergent.deck import N_CALLS, call_name, owners_to_pbn

T_MAX = 48                      # the belief net reads at most 48 calls
HCP_BINS = 31
CARD_HCP = torch.zeros(52)
for _s in range(4):
    CARD_HCP[_s * 13:_s * 13 + 4] = torch.tensor([4., 3., 2., 1.])
MAX_CALLS = 320                 # a rollout longer than any legal auction is a bug


def log(msg):
    print(f"bid search: {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ belief net

class BeliefMLP:
    """The research BeliefMLP (train.py) at inference: fp16 weights, fp32 maths.

    Input: own hand (52), vulnerability ours/theirs (2), dealer relative to the viewer
    (4), a system token per side (one-hot over n_sys + 1, the last = not told), and a
    one-hot (position, relative seat, call) grid of the auction. Outputs: owner logits
    (n, 52, 3) over LHO / partner / RHO, and the summary heads (n, 3 * (31 + 4 * 14)).
    """

    def __init__(self, state_dict, n_sys, layers, half=True):
        dt = torch.float16 if half else torch.float32
        self.w = {k: v.to(dt).contiguous() for k, v in state_dict.items() if not k.startswith("sys_head")}
        self.n_sys = n_sys
        self.layers = layers

    @classmethod
    def load(cls, path, half=True):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "n_sys" in ck:                                    # the server's copy (sync_models.sh)
            return cls(ck["net"], ck["n_sys"], ck["layers"], half), ck
        sd = ck["net"]                                       # a research checkpoint, as is
        return cls(sd, sd["sys_head.weight"].shape[0] // 2, ck["args"]["layers"], half), ck

    def _lin(self, x, name):
        return F.linear(x, self.w[name + ".weight"].float(), self.w[name + ".bias"].float())

    def _ln(self, x, name):
        return F.layer_norm(x, x.shape[-1:], self.w[name + ".weight"].float(), self.w[name + ".bias"].float())

    @torch.no_grad()
    def __call__(self, b):
        n, m = b["calls"].shape
        grid = torch.zeros(n, T_MAX * 4 * (N_CALLS + 1))
        pos = torch.arange(m)[None]
        grid.scatter_(1, (pos * 4 + b["seat"]) * (N_CALLS + 1) + b["calls"], b["live"].float())
        sys_tok = b["sys_in"].clamp(max=self.n_sys)
        x = torch.cat([b["hand"], b["ctx"], F.one_hot(b["dealer"], 4).float(),
                       F.one_hot(sys_tok, self.n_sys + 1).flatten(1).float(), grid], 1)
        cols = torch.nonzero(x.abs().sum(0) > 0).squeeze(1)  # x is almost all zeros
        h = x[:, cols] @ self.w["inp.weight"][:, cols].float().T + self.w["inp.bias"].float()
        for i in range(self.layers):
            p = f"blocks.{i}"
            h = h + self._lin(F.gelu(self._lin(self._ln(h, f"{p}.0"), f"{p}.1")), f"{p}.3")
        h = self._ln(h, "norm")
        return self._lin(h, "owner").view(n, 52, 3), self._lin(h, "sum_head")


def belief_input(hand, calls, dealer, vul, n_sys):
    """One-row batch for the seat on turn (the viewer), as train.make_batch builds it."""
    t = len(calls)
    v = (dealer + t) % 4
    m = max(1, t)
    k = torch.arange(m)[None]
    live = k < t
    hist = torch.tensor([list(calls)]) if t else torch.zeros(1, 1, dtype=torch.long)
    calls_t = torch.where(live, hist, torch.full_like(hist, N_CALLS))
    ours, theirs = (vul[0], vul[1]) if v % 2 == 0 else (vul[1], vul[0])
    return {"hand": torch.as_tensor(np.asarray(hand), dtype=torch.float32).view(1, 52),
            "ctx": torch.tensor([[float(ours), float(theirs)]]),
            "dealer": torch.tensor([(dealer - v) % 4]),
            "calls": calls_t, "seat": (dealer + k - v) % 4, "live": live,
            "sys_in": torch.tensor([[n_sys, n_sys]])}, v


def sinkhorn(owner_logits, hand, iters=10):
    """Rescale to 13 cards per hidden seat; own cards get prob 0. Returns log-probs (n,52,3)."""
    lp = owner_logits.log_softmax(-1)
    hidden = (hand < 0.5)[..., None]
    for _ in range(iters):
        p = lp.exp() * hidden
        col = p.sum(1, keepdim=True).clamp(min=1e-6)
        lp = lp + (math.log(13.0) - col.log())
        lp = lp.log_softmax(-1)
    return lp


def split_summary(raw):
    """raw (n, 261) -> hcp logits (n,3,31), length logits (n,3,4,14)."""
    r = raw.view(-1, 3, HCP_BINS + 56)
    return r[..., :HCP_BINS], r[..., HCP_BINS:].reshape(-1, 3, 4, 14)


# ------------------------------------------------------------------ shape-first sampler
# sample_eval.py's shape / draw_shapes / deal_shapes, kept call for call so the same
# generator seed draws the same deals.

def summarize(rel):
    """rel (B,M,52) relative owners -> hcp (B,M,3), lengths (B,M,3,4) of LHO/partner/RHO."""
    one = torch.stack([(rel == j) for j in (1, 2, 3)], -1).float()
    hcp = (one * CARD_HCP[None, None, :, None]).sum(2)
    lens = one.view(*one.shape[:2], 4, 13, 3).sum(3).transpose(2, 3)
    return hcp, lens


def resample(rel, logw, n, g):
    pick = torch.multinomial(torch.softmax(logw, 1), n, replacement=True, generator=g)
    return rel.gather(1, pick[..., None].expand(-1, -1, 52))


def draw_shapes(len_logits, hand, m, g, cand=4000, tries=4):
    """Shapes (B,m,3,4) for LHO/partner/RHO from the length heads, each suit summing to its
    hidden cards and each hand to 13 (exact rejection; a repair step if nothing passes)."""
    b = len_logits.shape[0]
    p = len_logits.softmax(-1)
    t = 13 - hand.view(b, 4, 13).sum(-1).round().long()
    l = torch.arange(14)
    out = torch.zeros(b, m, 3, 4, dtype=torch.long)
    todo = list(range(b))
    best = {}
    for _ in range(tries):
        if not todo:
            break
        idx = torch.tensor(todo)
        L = torch.zeros(len(idx), cand, 3, 4, dtype=torch.long)
        for s in range(4):
            ps = p[idx, :, s]
            l3 = t[idx, s][:, None, None] - l[None, :, None] - l[None, None, :]
            ok = (l3 >= 0) & (l3 <= 13)
            w = ps[:, 0, :, None] * ps[:, 1, None, :] * ps[:, 2].gather(1, l3.clamp(0, 13).view(len(idx), -1)).view(-1, 14, 14)
            w = (w * ok).view(len(idx), -1)
            w = torch.where(w.sum(1, keepdim=True) > 0, w, ok.view(len(idx), -1).float())
            k = torch.multinomial(w, cand, replacement=True, generator=g)
            L[:, :, 0, s], L[:, :, 1, s] = k // 14, k % 14
            L[:, :, 2, s] = t[idx, s][:, None] - L[:, :, 0, s] - L[:, :, 1, s]
        dev = (L.sum(-1) - 13).abs().sum(-1)
        left = []
        for i, r in enumerate(todo):
            good = torch.nonzero(dev[i] == 0).squeeze(1)
            if len(good):
                out[r] = L[i, good[torch.randint(len(good), (m,), generator=g)]]
            else:
                j = int(dev[i].argmin())
                if r not in best or dev[i, j] < best[r][0]:
                    best[r] = (int(dev[i, j]), L[i, j].clone())
                left.append(r)
        todo = left
    for r in todo:                                   # repair: move cards between hands
        S = best[r][1]
        while True:
            tot = S.sum(-1)
            hi, lo = int(tot.argmax()), int(tot.argmin())
            if tot[hi] == 13:
                break
            s = int(torch.nonzero(S[hi] > 0)[0])
            S[hi, s] -= 1
            S[lo, s] += 1
        out[r] = S
        log(f"shape fallback (best deviation {best[r][0]})")
    return out


def deal_shapes(prob, hand, L, g):
    """prob (B,52,3), shapes (B,m,3,4) -> (B,m,52) relative owners: honours by the owner
    probabilities among hands with room in the suit, small cards at random."""
    b, m = L.shape[:2]
    hid = hand < 0.5
    rel = torch.zeros(b, m, 52, dtype=torch.long)
    for s in range(4):
        cap = L[..., s].clone().float()
        for r in torch.randperm(4, generator=g).tolist():
            c = s * 13 + r
            w = prob[:, None, c, :] * (cap > 0)
            w = torch.where(w.sum(-1, keepdim=True) > 0, w, (cap > 0).float())
            w = torch.where(w.sum(-1, keepdim=True) > 0, w, torch.ones_like(w))
            pick = torch.multinomial(w.view(-1, 3), 1, generator=g).view(b, m)
            h = hid[:, c][:, None].expand(-1, m)
            rel[..., c] = torch.where(h, pick + 1, 0)
            cap -= F.one_hot(pick, 3).float() * h[..., None]
        small = torch.arange(s * 13 + 4, s * 13 + 13)
        keys = torch.rand(b, m, 9, generator=g)
        keys[(~hid[:, small])[:, None].expand(-1, m, -1)] = 2.0
        rank = keys.argsort(-1).argsort(-1)
        c1, c2, c3 = cap[..., 0:1], cap[..., 0:2].sum(-1, keepdim=True), cap.sum(-1, keepdim=True)
        rel[..., small] = torch.where(rank < c1, 1, torch.where(rank < c2, 2, torch.where(rank < c3, 3, 0)))
    return rel


def shape_sample(prob, hand, summary, n, g, m=200):
    """Shape-first: m shapes, honours by belief, small cards uniform; n resampled by HCP head."""
    hl, ll = split_summary(summary)
    L = draw_shapes(ll, hand, m, g)
    rel = deal_shapes(prob, hand, L, g)
    hcp, _ = summarize(rel)
    hi = hcp.long().clamp(max=HCP_BINS - 1)
    logw = hl.log_softmax(-1)[:, None].expand(-1, m, -1, -1).gather(-1, hi[..., None]).squeeze(-1).sum(-1)
    return resample(rel, logw, n, g)


@torch.no_grad()
def sample_deals(net, hand, calls, dealer, vul, n, g):
    """(n, 52) absolute owners (0..3 = N E S W), the viewer's own cards where they are."""
    b, v = belief_input(hand, calls, dealer, vul, net.n_sys)
    owner_logits, summary = net(b)
    prob = sinkhorn(owner_logits.float(), b["hand"]).exp()
    rel = shape_sample(prob, b["hand"], summary.float(), n, g)[0]
    assert ((b["hand"][0] > 0.5) == (rel == 0).all(0)).all(), "viewer hand not fixed"
    return ((rel + v) % 4).numpy()


# ------------------------------------------------------------------ rollouts and scoring

@torch.no_grad()
def rollout(bot, owners, prefix, firsts, dealer, vul):
    """Finished AuctionStates, one per (first call, deal): `firsts` x `owners` rows in that
    order. Every seat is `bot`, greedy, holding the deal's cards."""
    from emergent import engine
    st0 = AuctionState.from_calls(prefix, dealer=dealer, vul_ns=vul[0], vul_ew=vul[1])
    n = len(owners)
    hands = torch.zeros(n, 4, 52)
    hands[torch.arange(n)[:, None], torch.as_tensor(owners), torch.arange(52)[None]] = 1.0
    states = [st0.apply(int(c)) for c in firsts for _ in range(n)]
    deal_of = [i for _ in firsts for i in range(n)]
    vns, vew = torch.tensor([float(vul[0])]), torch.tensor([float(vul[1])])
    while True:
        live = [i for i, s in enumerate(states) if not s.ended]
        if not live:
            return states
        if len(states[live[0]].calls) >= MAX_CALLS:
            raise RuntimeError("rollout did not end")
        r = len(live)
        hist = torch.tensor([states[i].calls for i in live], dtype=torch.long)
        seat = torch.tensor([states[i].turn for i in live])
        legal = torch.tensor([engine.legal_calls(bot, states[i]) for i in live])
        lp = bot.batch_log_probs(hands[[deal_of[i] for i in live], seat], hist,
                                 torch.full((r,), dealer), vns.expand(r), vew.expand(r), seat, legal)
        for i, c in zip(live, lp.argmax(-1).tolist()):
            states[i] = states[i].apply(int(c))


# endplay's Denom: spades 0, hearts 1, diamonds 2, clubs 3, nt 4; bids count C D H S NT
_DENOM_OF_STRAIN = (3, 2, 1, 0, 4)


def dd_tricks(owners, declarer, strain):
    """Double dummy tricks for `declarer` (0..3) in bid strain `strain` (C D H S NT)."""
    from endplay.dds import analyse_start
    from endplay.types import Deal, Denom, Player
    deal = Deal.from_pbn(owners_to_pbn(np.asarray(owners)))
    deal.trump = Denom(_DENOM_OF_STRAIN[strain])
    deal.first = Player((declarer + 1) % 4)
    return int(analyse_start(deal))


def side_score(st, tricks_of, side):
    """Score of a finished auction for `side` (0 NS, 1 EW); tricks_of(declarer, strain)."""
    if st.last_contract < 0:
        return 0
    _, level, strain = CONTRACTS[st.last_contract]
    decl = st.declarer()
    raw = contract_score(level, strain, tricks_of(decl, strain), st.doubled, st.vulnerable(decl % 2))
    return raw if decl % 2 == side else -raw


# ------------------------------------------------------------------ the search

class BidSearch:
    """One searcher per process: the belief net plus a cache of solved contracts."""

    def __init__(self, net, samples=32, k=3, margin=50.0, budget_ms=2400.0, pmin=0.02,
                 min_samples=8, dd_lock=None):
        self.net = net
        self.samples, self.k, self.margin = samples, k, margin
        self.budget_ms, self.pmin, self.min_samples = budget_ms, pmin, min_samples
        self.dd_lock = dd_lock or threading.Lock()
        self.cache = {}                 # (owners bytes, declarer, strain) -> tricks
        self.searches = 0               # searches that ran (more than one candidate)
        self.solves = 0                 # DD solves in the last search

    def candidates(self, probs, legal):
        """Greedy first, then the next best legal calls with p >= pmin, K in all."""
        p = torch.as_tensor(probs).masked_fill(~torch.as_tensor(legal), -1.0)
        greedy = int(p.argmax())
        order = p.argsort(descending=True)[:self.k].tolist()
        return [greedy] + [a for a in order if a != greedy and float(p[a]) >= self.pmin]

    def _tricks(self, owners, key, declarer, strain):
        ck = (key, declarer, strain)
        if ck not in self.cache:
            if len(self.cache) > 200_000:
                self.cache.clear()
            with self.dd_lock:
                self.cache[ck] = dd_tricks(owners, declarer, strain)
            self.solves += 1
        return self.cache[ck]

    @staticmethod
    def seed_of(hand, calls, dealer, vul):
        """The position as a seed: the same position draws the same deals."""
        key = np.asarray(hand, np.uint8).tobytes() + bytes(list(calls)) + bytes([dealer, *map(int, vul)])
        return zlib.crc32(key)

    @torch.no_grad()
    def choose(self, bot, hand, calls, dealer, vul, probs, legal, g=None, budget_ms="config"):
        """(call, info). `probs`/`legal`: the bot's own policy and legal calls here.
        info is None when nothing was searched (one candidate), else what happened."""
        t0 = time.perf_counter()
        budget = self.budget_ms if budget_ms == "config" else budget_ms
        cands = self.candidates(probs, legal)
        greedy = cands[0]
        if len(cands) == 1:
            return greedy, None
        info = {"cands": cands, "greedy": greedy, "pick": greedy, "used": 0}
        if len(calls) >= T_MAX:
            info["skip"] = "auction longer than the belief net reads"
            return greedy, info
        self.searches += 1
        self.solves = 0
        if g is None:
            g = torch.Generator().manual_seed(self.seed_of(hand, calls, dealer, vul))
        owners = sample_deals(self.net, hand, calls, dealer, vul, self.samples, g)
        t_sample = time.perf_counter()
        states = rollout(bot, owners, list(calls), cands, dealer, vul)
        t_roll = time.perf_counter()
        side = (dealer + len(calls)) % 2
        n, k = len(owners), len(cands)
        scores = np.zeros((k, n))
        used = 0
        for i in range(n):
            key = owners[i].astype(np.uint8).tobytes()
            try:
                for j in range(k):
                    if budget is not None and (time.perf_counter() - t0) * 1000 > budget:
                        raise TimeoutError
                    scores[j, i] = side_score(states[j * n + i],
                                              lambda d, s: self._tricks(owners[i], key, d, s), side)
            except TimeoutError:
                break
            used += 1
        means = scores[:, :used].mean(1) if used else np.zeros(k)
        info.update(used=used, means=means.round(1).tolist(), solves=self.solves,
                    distinct=len({o.tobytes() for o in owners[:used]}),
                    ms={"sample": round((t_sample - t0) * 1000), "rollout": round((t_roll - t_sample) * 1000),
                        "total": round((time.perf_counter() - t0) * 1000)})
        if used < min(self.min_samples, n):
            info["skip"] = f"over budget: {used} of {n} deals solved"
            return greedy, info
        best = int(means.argmax())
        pick = cands[best] if means[best] >= means[0] + self.margin else greedy
        info["pick"] = pick
        return pick, info


def describe(calls, dealer, info):
    """One log line for a search."""
    seat = "NESW"[(dealer + len(calls)) % 4]
    head = f"{seat} after '{' '.join(call_name(c) for c in calls) or '-'}'"
    if "means" not in info:
        return f"{head}: {info.get('skip')} -> {call_name(info['pick'])}"
    cands = ", ".join(f"{call_name(c)} {m:+.0f}" for c, m in zip(info["cands"], info["means"]))
    what = info.get("skip") or ("DEVIATE" if info["pick"] != info["greedy"] else "net's call")
    return (f"{head}: [{cands}] on {info['used']} deals ({info['distinct']} distinct, {info['solves']} DD solves), "
            f"{info['ms']['total']} ms (sample {info['ms']['sample']}, rollout {info['ms']['rollout']}) "
            f"-> {call_name(info['pick'])} ({what})")


__all__ = ["BeliefMLP", "BidSearch", "describe", "dd_tricks", "rollout", "sample_deals"]
