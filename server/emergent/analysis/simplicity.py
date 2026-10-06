"""How simple is a model's bidding? Counts the "code words" in its self-play auctions.

A code word is a call partner cannot read at face value: a suit bid without length
in that suit (4+ cards, or 3+ to raise partner's suit), a low double (the contract is
at level 3 or below, so it is takeout-style, not penalty), a redouble, or a strong
artificial 2♣ opening. Fewer code words = easier for a beginner to follow.

Reference values (10,000 self-play boards): E46 14, E44 17, EPBot Acol 63, SAYC 75,
2/1 76, Polish Club 83, Precision 92 code words per 100 auctions.

    python tools/simplicity.py runs/training/3_table/best.pt --boards 4000
    python tools/simplicity.py --boards-npz results/x/boards.npz   # analyse existing auctions
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.bridge.calls import ACTION_NAMES  # noqa: E402
from training.bridge.deals import load_dataset  # noqa: E402

DATA = "data/dds_results_100M.npy"
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
    tables = ([("hist1", None)] if who is None else
              [("hist1", 0 if who == "A" else 1), ("hist2", 1 if who == "A" else 0)])
    c = dict.fromkeys(("auctions", "bids", "suit_bids", "natural", "cue", "jumps", "bw",
                       "strong2c", "x_low", "xx", "side_calls", "competitive", "calls"), 0)
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
            if name == "P":
                continue
            side_calls[p % 2] += 1
            if name == "X":
                c["x_low"] += mine and cur_level <= 3
                continue
            if name == "XX":
                c["xx"] += mine
                continue
            level, strain = int(name[0]), name[1:]
            rank = RANK[strain]
            c["bids"] += mine
            if not opened:
                c["strong2c"] += mine and name == "2C" and (lens[p][3] < 5 or hcp[p] >= 20)
                opened = True
            c["jumps"] += mine and level > (max(cur_level, 1) if rank > cur_rank else cur_level + 1)
            c["bw"] += mine and name == "4NT"
            if strain in SUIT:
                k, partner = SUIT[strain], (p + 2) % 4
                if mine:
                    c["suit_bids"] += 1
                    c["natural"] += lens[p][k] >= 4 or (strain in bid_suits[partner] and lens[p][k] >= 3)
                    c["cue"] += (any(strain in bid_suits[q] for q in ((p + 1) % 4, (p + 3) % 4))
                                 and strain not in bid_suits[p] | bid_suits[partner])
                bid_suits[p].add(strain)
            cur_level, cur_rank, last_bidder = level, rank, p
        if last_bidder is not None:
            c["side_calls"] += side_calls[last_bidder % 2]
            c["competitive"] += side_calls[1 - last_bidder % 2] > 0
    n = max(c["auctions"], 1)
    return {
        "code_words_per_100": 100 * ((c["suit_bids"] - c["natural"]) + c["x_low"] + c["xx"]
                                     + c["strong2c"]) / n,
        "natural_suit_bids": c["natural"] / max(c["suit_bids"], 1),
        "cue_bids_per_100": 100 * c["cue"] / n,
        "low_doubles_per_100": 100 * c["x_low"] / n,
        "redoubles_per_100": 100 * c["xx"] / n,
        "four_nt_per_100": 100 * c["bw"] / n,
        "jump_share": c["jumps"] / max(c["bids"], 1),
        "calls_per_auction": c["calls"] / n,
        "declaring_side_calls": c["side_calls"] / n,
        "both_sides_bid": c["competitive"] / n,
        "auctions": c["auctions"],
    }


def self_play(checkpoint: Path, boards: int, threads: int, data: str = DATA) -> dict:
    """Play ``checkpoint`` against itself with tools/match.py and analyse the auctions."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "tools" / "match.py"),
                        "--a", f"four:{checkpoint}", "--b", f"four:{checkpoint}",
                        "--boards", str(boards), "--threads", str(threads),
                        "--data", data, "--out", tmp],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        return analyse(dict(np.load(Path(tmp) / "boards.npz")), data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--boards-npz", help="analyse an existing match.py boards.npz instead")
    p.add_argument("--boards", type=int, default=4000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--data", default=DATA)
    args = p.parse_args()
    if args.boards_npz:
        result = analyse(dict(np.load(args.boards_npz)), args.data)
    elif args.checkpoint:
        result = self_play(Path(args.checkpoint).resolve(), args.boards, args.threads, args.data)
    else:
        p.error("give a checkpoint or --boards-npz")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
