"""The bot engine: how the server's bots choose a call and a card.

Every surface asks here: the bid desk (/), the play desk (/play), the table (/table)
and the machine APIs (/apis/bbo.php, /apis/brill/*). So a bot bids and plays the
same way wherever it sits, and `CONFIG` is the one place its behaviour is set. It is
read from the environment once, at import. A page may override search for one game
(the /table toggle); nothing else is overridden anywhere.

The bots themselves are the desks' classes (`bidserver.FourSeatBot` and friends,
`playdesk.PlayBot`), registered in `BID_MODELS` / `PLAY_MODELS` by their loaders.
A bidding bot's `decide` is its one forward pass; its desk `view` draws from the
same pass, so the call the page shows is the call the bot makes.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from training.bridge.auction import AuctionState
from training.bridge.calls import DOUBLE, PASS, REDOUBLE
from training.fourseat.state import features_from_history
from training.play.model import encode
from emergent.deck import N_CALLS


@dataclass(frozen=True)
class Config:
    """How the bots behave. Defaults are production's; each has an env override."""

    bid_model: str | None = None      # BOT_BID_MODEL; None = first entry of models/models.json
    play_model: str | None = None     # BOT_PLAY_MODEL; None = first of models/play_models.json
    # PIMC for card play. 20 layouts is where the offline gain flattens (+0.99 IMP a
    # board over the plain net); the budget is what keeps the opening lead bearable on
    # a one-core box, where a trick-one solve costs ~290 ms against a laptop's 6 ms.
    search: bool = True               # PLAY_SEARCH=0 turns it off by default
    samples: int = 20                 # PLAY_SEARCH_SAMPLES
    budget_ms: float = 900.0          # PLAY_SEARCH_BUDGET_MS
    # Declarer always searches. Defence searches too, but not before trick index 2
    # (the third trick). Measured on 600 boards, that gives the same defence regret as
    # searching every turn (0.537) for 45% of the cost -- tricks 0 and 1 add nothing,
    # and on this box they are nearly all of the price.
    defence: str = "all"              # PLAY_SEARCH_DEFENCE: off / lead / all / only
    defence_from: int = 2             # PLAY_SEARCH_DEFENCE_FROM
    # Bidding search (emergent/bidsearch.py), /debug only and only with the page's search
    # toggle on. The research settings: 32 belief-sampled deals, the net's top 3 calls,
    # leave its call only for a 50-point gain. A DD solve costs 30-500 ms on the droplet,
    # so the budget, not the sample count, usually decides how many deals are scored;
    # under MIN_SAMPLES solved in time the net's own call stands.
    bid_samples: int = 32             # BID_SEARCH_SAMPLES
    bid_k: int = 3                    # BID_SEARCH_K (1 = never search)
    bid_margin: float = 50.0          # BID_SEARCH_MARGIN, points
    bid_budget_ms: float = 2400.0     # BID_SEARCH_BUDGET_MS, wall clock per call
    bid_pmin: float = 0.02            # BID_SEARCH_PMIN, smallest policy p of a candidate
    bid_min_samples: int = 8          # BID_SEARCH_MIN_SAMPLES
    belief: str = "belief_r2.pt"      # BID_SEARCH_BELIEF, in models/

    @classmethod
    def from_env(cls, env=os.environ):
        d = cls()
        return cls(bid_model=env.get("BOT_BID_MODEL") or None,
                   play_model=env.get("BOT_PLAY_MODEL") or None,
                   search=env.get("PLAY_SEARCH", "1") not in ("0", "false", "off"),
                   samples=int(env.get("PLAY_SEARCH_SAMPLES", d.samples)),
                   budget_ms=float(env.get("PLAY_SEARCH_BUDGET_MS", d.budget_ms)),
                   defence=env.get("PLAY_SEARCH_DEFENCE", d.defence),
                   defence_from=int(env.get("PLAY_SEARCH_DEFENCE_FROM", d.defence_from)),
                   bid_samples=int(env.get("BID_SEARCH_SAMPLES", d.bid_samples)),
                   bid_k=int(env.get("BID_SEARCH_K", d.bid_k)),
                   bid_margin=float(env.get("BID_SEARCH_MARGIN", d.bid_margin)),
                   bid_budget_ms=float(env.get("BID_SEARCH_BUDGET_MS", d.bid_budget_ms)),
                   bid_pmin=float(env.get("BID_SEARCH_PMIN", d.bid_pmin)),
                   bid_min_samples=int(env.get("BID_SEARCH_MIN_SAMPLES", d.bid_min_samples)),
                   belief=env.get("BID_SEARCH_BELIEF", d.belief))


CONFIG = Config.from_env()

BID_MODELS = OrderedDict()    # id -> bidding bot, filled by bidserver.load_models
PLAY_MODELS = OrderedDict()   # id -> PlayBot, filled by playdesk.load_models
# One searcher per play bot, and it keeps the auction between `start` and `choose`:
# every caller, from any page or API thread, takes this lock around the pair.
SEARCH_LOCK = threading.Lock()


def default_bid_model():
    """The bidding model id a page or API uses when none is asked for."""
    if CONFIG.bid_model in BID_MODELS:
        return CONFIG.bid_model
    return next(iter(BID_MODELS), None)


def default_play_model():
    if CONFIG.play_model in PLAY_MODELS:
        return CONFIG.play_model
    return next(iter(PLAY_MODELS), None)


def search_on(search=None):
    """A page's search setting, None meaning "the config's"."""
    return CONFIG.search if search is None else bool(search)


# ------------------------------------------------------------------ bidding

def legal_calls(bot, st):
    """38 bools: the rules at `st`, narrowed to the calls this bidding model can make."""
    mask = st.legal_mask()[:N_CALLS].tolist()
    if not bot.doubles:
        mask[DOUBLE] = False
    if not bot.redouble:
        mask[REDOUBLE] = False
    if bot.final_only and not (st.last_contract >= 0 and st.pass_count == 2):
        mask[DOUBLE] = mask[REDOUBLE] = False
    return mask


def fourseat_features(bot, history, dealer, vul_ns, vul_ew, actor):
    """(B, width) input of a training four-seat net; `history` (B, T) with -1 padding.

    Competitive (E28, D5OWN4XC) nets read two more bits: [149] Pass would end the
    auction, [150] the contract stands redoubled. A copy of fourseat/competitive.py,
    whose imports pull in the trainer.
    """
    feats = features_from_history(history, dealer, vul_ns, vul_ew, actor, bot.doubles)
    if not bot.competitive:
        return feats
    extra = torch.zeros(len(history), 2)
    if history.shape[1]:
        pos = torch.arange(history.shape[1])[None]
        none = torch.full_like(history, -1)
        valid = history >= 0
        last_bid = torch.where(valid & (history < PASS), pos, none).max(1).values
        last_active = torch.where(valid & (history != PASS), pos, none).max(1).values
        last_xx = torch.where(history == REDOUBLE, pos, none).max(1).values
        trailing = valid.sum(1) - 1 - last_active
        extra[:, 0] = ((last_bid >= 0) & (trailing == 2)).float()
        extra[:, 1] = (last_xx > last_bid).float()
    return torch.cat((feats, extra), 1)


@torch.no_grad()
def choose_call(bot, hand, calls, dealer=0, vul=(False, False)):
    """(call, top) for the seat on turn: the bot's call, and its best four legal calls
    as [(call, p)] (Q instead of p for a net without a policy). Greedy, as in every match.

    `hand` is the caller's 52 card bits; `calls` a legal, unfinished auction from
    `dealer`. The desks use dealer North, nobody vulnerable; the APIs the real ones.
    """
    st = AuctionState.from_calls(list(calls), dealer=dealer)
    if st.ended:
        raise ValueError("the auction is already over")
    legal = legal_calls(bot, st)
    hand = torch.as_tensor(np.asarray(hand), dtype=torch.float32)[None]
    d = bot.decide(hand, list(calls), dealer, vul, legal)
    score = d["policy"] or d["q"]
    order = sorted((c for c in range(N_CALLS) if legal[c]), key=lambda c: -score[c])
    return d["pick"], [(c, float(score[c])) for c in order[:4]]


# ------------------------------------------------------------------ bidding search

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_BID_SEARCH = {}              # "searcher": the one BidSearch, built on first use
_BID_SEARCH_LOCK = threading.Lock()


def bid_searcher():
    """The bidding searcher, its belief net loaded on first use (32 MB that only /debug
    with search on needs). None, logged once, when the belief net is missing."""
    with _BID_SEARCH_LOCK:
        if "searcher" not in _BID_SEARCH:
            from emergent import bidsearch
            path = os.path.join(MODELS_DIR, CONFIG.belief)
            try:
                net, ck = bidsearch.BeliefMLP.load(path)
                _BID_SEARCH["searcher"] = bidsearch.BidSearch(
                    net, CONFIG.bid_samples, CONFIG.bid_k, CONFIG.bid_margin, CONFIG.bid_budget_ms,
                    CONFIG.bid_pmin, CONFIG.bid_min_samples, dd_lock=SEARCH_LOCK)
                bidsearch.log(f"belief net {CONFIG.belief} (step {ck.get('step')}) loaded; {CONFIG}")
            except Exception as exc:                 # noqa: BLE001 -- the page still bids
                bidsearch.log(f"no belief net ({exc!r}): the nets bid without search")
                _BID_SEARCH["searcher"] = None
        return _BID_SEARCH["searcher"]


def search_call(bot, hand, calls, dealer=0, vul=(False, False)):
    """The bidding search's call for the seat on turn (/debug with search on).

    The bot's own greedy call whenever the search cannot help: one candidate, no belief
    net, over budget, or any error. Every search writes one line to the log."""
    from emergent import bidsearch
    st = AuctionState.from_calls(list(calls), dealer=dealer)
    if st.ended:
        raise ValueError("the auction is already over")
    legal = legal_calls(bot, st)
    d = bot.decide(torch.as_tensor(np.asarray(hand), dtype=torch.float32)[None], list(calls), dealer, vul, legal)
    if CONFIG.bid_k <= 1 or not hasattr(bot, "batch_log_probs"):
        return d["pick"]
    searcher = bid_searcher()
    if searcher is None:
        return d["pick"]
    try:
        call, info = searcher.choose(bot, hand, list(calls), dealer, vul, d["policy"], legal)
    except Exception as exc:                         # noqa: BLE001 -- fall back, never fail a bid
        bidsearch.log(f"error ({exc!r}); the net's call {d['pick']} stands")
        return d["pick"]
    if info is not None:
        bidsearch.log(f"[{getattr(bot, 'id', '?')}] " + bidsearch.describe(calls, dealer, info))
    return call


# ------------------------------------------------------------------ card play

def searcher(bot):
    """PIMC over this play bot's own net, built on first use.

    Built late because the solver's tables cost ~120 MB and most requests never
    search. The budget matters more than the sample count here: a solve is ~300x
    dearer at trick one than at trick seven, so a fixed count would stall the
    opening and waste effort at the end.
    """
    if getattr(bot, "_searcher", None) is None:
        from training.play.search import PIMCPlayer
        bot._searcher = PIMCPlayer.from_net(bot.net, CONFIG.samples, budget_ms=CONFIG.budget_ms,
                                            defence=CONFIG.defence,
                                            defence_from_trick=CONFIG.defence_from)
    return bot._searcher


@torch.no_grad()
def card_policy(bot, contracts, batch):
    """The play net's forward pass for the seat on turn: (probs over 52, outputs, encoding)."""
    auction = bot.net.auction(contracts.calls, contracts.n_calls, contracts.dealer)
    enc = encode(batch, contracts, batch.t > 0, auction)
    out = bot.net(enc["features"], batch.legal())
    return out["log_probs"][0].exp(), out, enc


def seat_free_sampler(seed):
    """A layout sampler that numbers the hidden seats from the seat on turn.

    The stock sampler walks the seats as N, E, S, W, so the same dice deal a hidden card
    to a different opponent once the whole deal is turned round the table. Walking them
    from the seat on turn makes a turned deal draw the same layouts, turned with it.
    """
    from dataclasses import replace

    from training.play.search import LayoutSampler

    class SeatFree(LayoutSampler):
        def draw(self, position):
            def rel(s):
                return (s - position.turn) % 4
            return super().draw(replace(
                position,
                caps={s: position.caps[s] for s in sorted(position.caps, key=rel)},
                suit_seats=[sorted(seats, key=rel) for seats in position.suit_seats]))

    return SeatFree(seed)


@torch.no_grad()
def choose_card(bot, contracts, batch, search=None, seed=None):
    """(card, top) for the seat on turn at `batch`: the card, and the net's own best
    four legal cards as [(card, p)].

    `search` None follows CONFIG. With search, declarer always searches and defence
    per CONFIG.defence / defence_from; a turn the searcher skips is the net's card.

    `seed` makes the search deterministic: fresh dice from `seed` (seat-free, see
    `seat_free_sampler`) and all CONFIG.samples layouts, with no clock to stop early.
    Without it the searcher's dice run on from the last request and the budget applies.
    """
    if getattr(bot, "family", None) == "playq":            # B2g: its own values, see playq.py
        return bot.choose_card(contracts, batch, search, seed)
    probs, _, _ = card_policy(bot, contracts, batch)
    legal = batch.legal()[0]
    card = int(probs.argmax())
    if search_on(search):
        with SEARCH_LOCK:
            player = searcher(bot)
            player.start(contracts)
            if seed is None:
                card = int(player.choose(batch, contracts, batch.legal(), batch.t)[0])
            else:
                kept = player.sampler, player.budget_ms
                player.sampler, player.budget_ms = seat_free_sampler(seed), None
                try:
                    card = int(player.choose(batch, contracts, batch.legal(), batch.t)[0])
                finally:
                    player.sampler, player.budget_ms = kept
    order = [int(c) for c in probs.argsort(descending=True) if legal[int(c)]]
    return card, [(c, float(probs[c])) for c in order[:4]]
