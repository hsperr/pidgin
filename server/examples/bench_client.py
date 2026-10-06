"""Example /bench client: plays a duplicate match between two of our models and posts it.

For your own bot, replace ``bid()`` with a call to it. The rest is the whole protocol:
get the deals, play each board twice (your bot N-S, then E-W), post the auctions.

    python3 -m emergent.bidserver &                      # or use the live site
    python3 examples/bench_client.py --bot D_cw_s75k --opp brl_fsp --boards 1000
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emergent import apis, bidserver, engine  # noqa: E402
from emergent.deck import call_token  # noqa: E402

SEATS = "NESW"


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def bid(bot, hand_text, calls, dealer, vul):
    """Your bot goes here: its hand (S.H.D.C), the calls so far from the dealer -> one call."""
    bits = np.zeros(52, np.float32)
    bits[apis.parse_hand(hand_text)] = 1
    call, _ = engine.choose_call(bot, bits, calls, dealer, vul)
    return call


def play(board, ns_bot, ew_bot):
    dealer = SEATS.index(board["dealer"])
    vul = apis.parse_vul(board["vul"])
    calls = []
    while not apis.AuctionState.from_calls(calls, dealer).ended:
        seat = (dealer + len(calls)) % 4
        bot = ns_bot if seat % 2 == 0 else ew_bot
        calls.append(bid(bot, board["hands"][SEATS[seat]], calls, dealer, vul))
    return " ".join(call_token(c) for c in calls)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8787")
    p.add_argument("--bot", default="D_cw_s75k")
    p.add_argument("--opp", default="brl_fsp")
    p.add_argument("--boards", type=int, default=1000)
    p.add_argument("--public", action="store_true")
    a = p.parse_args()
    bidserver.load_models()
    you, them = bidserver.MODELS[a.bot], bidserver.MODELS[a.opp]
    boards = []
    for offset in range(0, a.boards, 1000):
        boards += get(f"{a.url}/apis/bench/deals?offset={offset}&limit={min(1000, a.boards - offset)}")["boards"]
    rows = []
    for i, b in enumerate(boards):
        rows.append({"board": b["board"], "ns": play(b, you, them), "ew": play(b, them, you)})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(boards)} boards", flush=True)
    body = json.dumps({"bot": a.bot, "opponent": a.opp, "public": a.public, "boards": rows}).encode()
    req = urllib.request.Request(f"{a.url}/apis/bench/report", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        print(json.load(r))


if __name__ == "__main__":
    main()
