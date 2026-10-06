"""How simple is a model's bidding? Counts the "code words" in its self-play auctions.

A code word is a call partner cannot read at face value: a suit bid without length
in that suit (4+ cards, or 3+ to raise partner's suit), a low double (the contract is
at level 3 or below, so it is takeout-style, not penalty), a redouble, or a strong
artificial 2♣ opening. Fewer code words = easier for a beginner to follow.

    python tools/simplicity.py runs/bridgezero/4_D/best.pt --boards 4000
    python tools/simplicity.py --boards-npz results/x/boards.npz   # analyse existing auctions
    python tools/simplicity.py --boards-npz results/a_vs_brl/boards.npz --who B   # brl's calls
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

from bridgezero.simplicity import DATA, analyse  # noqa: E402,F401  (the one implementation)


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
    p.add_argument("--who", choices=("A", "B"),
                   help="with --boards-npz of a match between two bots: count only that "
                        "player's calls (e.g. B = brl in a model-vs-brl match)")
    args = p.parse_args()
    if args.boards_npz:
        result = analyse(dict(np.load(args.boards_npz)), args.data, who=args.who)
    elif args.checkpoint:
        result = self_play(Path(args.checkpoint).resolve(), args.boards, args.threads, args.data)
    else:
        p.error("give a checkpoint or --boards-npz")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
