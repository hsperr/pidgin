"""How simple is a model's bidding? The one place these numbers are computed.

A code word is a call partner cannot read at face value: a suit bid without length
in that suit (4+ cards, or 3+ to raise partner's suit), a low double (the contract is
at level 3 or below, so it is takeout-style, not penalty), a redouble, or a strong
artificial 2♣ opening. Fewer code words = easier for a beginner to follow.
Opening numbers (``opening_numbers``; code words do not catch these), from every hand that
could open (nobody had bid yet):
- ``light_open_share``: share of 0-7 HCP hands that open (destructive openings)
- ``unbalanced_nt_share``: share of 1NT/2NT openings without a balanced shape
- ``opening_hcp_spread``: HCP range (5th to 95th percentile) of an opening call, averaged
  over openings; wide = one call covers very different hands

Used by tools/simplicity.py, the trainer's validation log and the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .bridge.calls import ACTION_NAMES
from .bridge.deals import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = "data/dds_results_100M.npy"
LIGHT_HCP = 7                                     # "light" opening: this many HCP or fewer
BALANCED_SHAPES = ("4333", "4432", "5332")
PASS_ACTION = ACTION_NAMES.index("P")
SPREAD_MIN_HANDS = 20                             # calls rarer than this have no spread


def opening_numbers(calls, hcp, balanced) -> dict:
    """Opening simplicity from hands that could open: ``calls`` (action ids, Pass = did not
    open), the hands' ``hcp`` and ``balanced`` flags."""
    calls, hcp, balanced = (np.asarray(x) for x in (calls, hcp, balanced))
    opened = calls != PASS_ACTION
    light = hcp <= LIGHT_HCP
    nt = opened & (calls % 5 == 4) & (calls < 10)                 # 1NT or 2NT
    spreads, weights = [], []
    for call in np.unique(calls[opened]):
        m = calls == call
        if m.sum() >= SPREAD_MIN_HANDS:
            lo, hi = np.percentile(hcp[m], [5, 95])
            spreads.append(hi - lo)
            weights.append(m.sum())
    return {"light_open_share": float(opened[light].mean()) if light.any() else 0.0,
            "unbalanced_nt_share": float((~balanced[nt]).mean()) if nt.any() else 0.0,
            "opening_hcp_spread": float(np.average(spreads, weights=weights)) if spreads else 0.0}
SUIT = {"S": 0, "H": 1, "D": 2, "C": 3}          # card // 13 order in the dataset
RANK = {"C": 0, "D": 1, "H": 2, "S": 3, "NT": 4}  # bidding order


def analyse(boards: dict, data: str = DATA, owners=None, who: str | None = None) -> dict:
    """Simplicity numbers for a match.py ``boards.npz``.

    ``who=None``: the table-1 auctions, all four seats (a self-play match).
    ``who="A"`` or ``"B"``: both tables, counting only that player's calls (a match
    between two different bots); per-auction rates are then per table.
    ``owners``: per-board card owners ``(boards, 52)``; by default they are read from
    ``data`` at ``boards["deal_index"]``.
    """
    if owners is None:
        dataset, _ = load_dataset(ROOT / data if not Path(data).is_absolute() else data)
        owners = [dataset[int(d)] for d in boards["deal_index"]]
    if who not in (None, "A", "B"):
        raise ValueError("who must be A, B, or None")
    tables = ([("hist1", None)] if who is None else
              [("hist1", 0 if who == "A" else 1), ("hist2", 1 if who == "A" else 0)])
    c = dict.fromkeys(("auctions", "bids", "suit_bids", "natural", "cue", "jumps", "bw",
                       "strong2c", "x_low", "xx", "side_calls", "competitive", "calls", "code_words"), 0)
    could_open = []                                  # (call, hcp, balanced) per possible opener
    suit_of = np.arange(52) // 13
    points = np.maximum(0, 4 - np.arange(52) % 13)
    for i, key, side in ((i, key, side) for key, side in tables
                         for i in range(len(boards["dealer"]))):
        o = np.asarray(owners[i])
        lens = np.array([[((o == s) & (suit_of == k)).sum() for k in range(4)] for s in range(4)])
        hcp = np.array([points[o == s].sum() for s in range(4)])
        dealer = int(boards["dealer"][i])
        hist = [int(x) for x in boards[key][i] if x >= 0]
        cur_level, cur_rank, last_bidder, opened = 0, -1, None, False
        bid_suits = [set() for _ in range(4)]
        side_calls = [0, 0]
        c["auctions"] += 1
        for j, action in enumerate(hist):
            p, name = (dealer + j) % 4, ACTION_NAMES[action]
            mine = side is None or p % 2 == side          # this call is counted
            c["calls"] += mine
            if mine and not opened:                       # this hand may open
                shape = "".join(map(str, sorted(lens[p], reverse=True)))
                could_open.append((action, hcp[p], shape in BALANCED_SHAPES))
            if name == "P":
                continue
            side_calls[p % 2] += 1
            if name == "X":
                c["x_low"] += mine and cur_level <= 3
                c["code_words"] += mine and cur_level <= 3
                continue
            if name == "XX":
                c["xx"] += mine
                c["code_words"] += mine
                continue
            level, strain = int(name[0]), name[1:]
            rank = RANK[strain]
            c["bids"] += mine
            artificial = not opened and name == "2C" and (lens[p][3] < 5 or hcp[p] >= 20)
            unnatural = False
            if not opened:
                c["strong2c"] += mine and artificial
                opened = True
            c["jumps"] += mine and level > (max(cur_level, 1) if rank > cur_rank else cur_level + 1)
            c["bw"] += mine and name == "4NT"
            if strain in SUIT:
                k, partner = SUIT[strain], (p + 2) % 4
                if mine:
                    c["suit_bids"] += 1
                    natural = lens[p][k] >= 4 or (strain in bid_suits[partner] and lens[p][k] >= 3)
                    c["natural"] += natural
                    unnatural = not natural
                    c["cue"] += (any(strain in bid_suits[q] for q in ((p + 1) % 4, (p + 3) % 4))
                                 and strain not in bid_suits[p] | bid_suits[partner])
                bid_suits[p].add(strain)
            c["code_words"] += mine and (unnatural or artificial)
            cur_level, cur_rank, last_bidder = level, rank, p
        if last_bidder is not None:
            c["side_calls"] += side_calls[last_bidder % 2]
            c["competitive"] += side_calls[1 - last_bidder % 2] > 0
    n = max(c["auctions"], 1)
    return {
        "code_words_per_100": 100 * c["code_words"] / n,
        "natural_suit_bids": c["natural"] / max(c["suit_bids"], 1),
        "cue_bids_per_100": 100 * c["cue"] / n,
        "low_doubles_per_100": 100 * c["x_low"] / n,
        "redoubles_per_100": 100 * c["xx"] / n,
        "four_nt_per_100": 100 * c["bw"] / n,
        "jump_share": c["jumps"] / max(c["bids"], 1),
        **opening_numbers(*(zip(*could_open) if could_open else ([], [], []))),
        "calls_per_auction": c["calls"] / n,
        "declaring_side_calls": c["side_calls"] / n,
        "both_sides_bid": c["competitive"] / n,
        "auctions": c["auctions"],
    }


def analyse_batch(batch, deals) -> dict:
    """``analyse`` for a finished four-seat ``batch`` on ``deals`` (TorchDeals), all seats."""
    hist = batch.history.cpu().numpy()
    owners = deals.hands[batch.deal].argmax(1).cpu().numpy()       # (B, 52) seat per card
    return analyse({"hist1": hist, "dealer": batch.dealer.cpu().numpy()}, owners=owners)
