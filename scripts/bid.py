"""Ask Pidgin for one call.

    python scripts/bid.py AKQ2.JT9.876.543                       # opening, dealer North
    python scripts/bid.py AKQ2.JT9.876.543 --auction "1H P" --model PidginV1
    python scripts/bid.py KJ5.AQ32.K4.QJ98 --dealer W --vul ew --auction "P 1S"

The hand belongs to the seat on turn: dealer plus the number of calls so far.
Prints the call and the model's four best legal calls with their probabilities.
"""
from __future__ import annotations

import argparse

import numpy as np

from pidgin import SEATS, call_name, engine, load_bidder, parse_call, parse_hand, parse_vul


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hand", help="spades.hearts.diamonds.clubs, e.g. AKQ2.JT9.876.543")
    ap.add_argument("--auction", default="", help='calls so far, e.g. "1H P 2C"')
    ap.add_argument("--dealer", default="N", choices=list(SEATS))
    ap.add_argument("--vul", default="none", choices=["none", "ns", "ew", "both"])
    ap.add_argument("--model", default="PidginV2", help="team id or checkpoint file")
    args = ap.parse_args()

    bot = load_bidder(args.model)
    calls = [parse_call(t) for t in args.auction.split()]
    dealer = SEATS.index(args.dealer)
    seat = SEATS[(dealer + len(calls)) % 4]
    call, top = engine.choose_call(bot, np.asarray(parse_hand(args.hand)), calls, dealer, parse_vul(args.vul))
    print(f"{args.model} as {seat}: {call_name(call)}")
    for c, p in top:
        print(f"  {call_name(c):>3}  {p:.3f}")


if __name__ == "__main__":
    main()
