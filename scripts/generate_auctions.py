"""Let a model bid N deals against itself and save the auctions.

    python scripts/generate_auctions.py --n 1000 --out auctions.npz
    python scripts/generate_auctions.py --n 1000000 --model PidginV1 --out v1_1M.npz
    python scripts/generate_auctions.py --n 100 --pbn auctions.txt          # readable text
    python scripts/generate_auctions.py --n 10000 --data data/dds_results_100M.npy --start -10000

Deals are random (seeded) unless --data names a DDS dataset; then the deals are taken from
it and their double-dummy tables are saved too. Dealer and vulnerability rotate like a
real board set (board i). All four seats are the same model and bid greedily, exactly as
the server's bots do.

The .npz holds: calls (N, T) int8 padded with -1 (0..34 = 1C..7NT, 35 Pass, 36 X, 37 XX),
owners (N, 52) uint8 (seat 0..3 = N E S W holding card suit*13+rank, A high, suits S H D C),
dealer (N,), vul (N, 2) for NS/EW, and tricks (N, 4, 5) when --data is given.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pidgin import SEATS, call_name, engine, load_bidder
from training.bridge.auction import AuctionState

# Standard board rotation: dealer N E S W; vulnerability none, NS, EW, both over 16 boards.
VUL16 = [(0, 0), (1, 0), (0, 1), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0),
         (0, 1), (1, 1), (0, 0), (1, 0), (1, 1), (0, 0), (1, 0), (0, 1)]


def deals(args):
    if args.data:
        from training.bridge.deals import load_dataset
        owners, tricks = load_dataset(args.data)
        start = args.start if args.start >= 0 else len(owners) + args.start
        sl = slice(start, start + args.n)
        return np.asarray(owners[sl], dtype=np.uint8), np.asarray(tricks[sl], dtype=np.uint8)
    rng = np.random.default_rng(args.seed)
    base = np.repeat(np.arange(4, dtype=np.uint8), 13)
    return np.stack([rng.permutation(base) for _ in range(args.n)]), None


@torch.no_grad()
def bid_batch(bot, owners, dealer, vul):
    """Greedy auctions for a batch of deals; one forward pass per round of calls."""
    b = len(owners)
    hands = torch.as_tensor((owners[:, None, :] == np.arange(4)[None, :, None]).astype(np.float32))
    states = [AuctionState.from_calls([], dealer=int(d)) for d in dealer]
    history = [[] for _ in range(b)]
    batched = hasattr(bot, "batch_log_probs")
    while True:
        live = [i for i in range(b) if not states[i].ended]
        if not live:
            break
        seat = torch.tensor([(int(dealer[i]) + len(history[i])) % 4 for i in live])
        if batched:
            t = max(len(history[i]) for i in live)
            hist = torch.full((len(live), t), -1, dtype=torch.long)
            for r, i in enumerate(live):
                hist[r, :len(history[i])] = torch.tensor(history[i], dtype=torch.long)
            legal = torch.tensor([engine.legal_calls(bot, states[i]) for i in live])
            lp = bot.batch_log_probs(hands[live, seat], hist, torch.as_tensor(dealer[live]).long(),
                                     torch.as_tensor(vul[live, 0]).bool(), torch.as_tensor(vul[live, 1]).bool(),
                                     seat, legal)
            picks = lp.argmax(1).tolist()
        else:  # a bot without a batched path (BRL): one call at a time
            picks = [engine.choose_call(bot, hands[i, s].numpy(), history[i], int(dealer[i]),
                                        (bool(vul[i, 0]), bool(vul[i, 1])))[0]
                     for i, s in zip(live, seat.tolist())]
        for i, c in zip(live, picks):
            history[i].append(int(c))
            states[i] = states[i].apply(int(c))
    return history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="number of deals")
    ap.add_argument("--model", default="PidginV2", help="team id or checkpoint file")
    ap.add_argument("--out", default="auctions.npz")
    ap.add_argument("--pbn", help="also write a readable text file (PBN deal + auction per line)")
    ap.add_argument("--data", help="DDS dataset (.npy/.npz); default: random deals")
    ap.add_argument("--start", type=int, default=-1000, help="first deal in --data (negative = from the end)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    bot = load_bidder(args.model)
    owners, tricks = deals(args)
    n = len(owners)
    board = np.arange(n)
    dealer = (board % 4).astype(np.int64)
    vul = np.array([VUL16[i % 16] for i in board], dtype=np.uint8)

    auctions, t0 = [], time.time()
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        auctions += bid_batch(bot, owners[s:e], dealer[s:e], vul[s:e])
        print(f"{e}/{n} auctions, {time.time() - t0:.0f}s", flush=True)

    calls = np.full((n, max(map(len, auctions))), -1, dtype=np.int8)
    for i, a in enumerate(auctions):
        calls[i, :len(a)] = a
    out = dict(calls=calls, owners=owners, dealer=dealer.astype(np.uint8), vul=vul, model=np.array(args.model))
    if tricks is not None:
        out["tricks"] = tricks
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, calls.shape)

    if args.pbn:
        from emergent.deck import owners_to_pbn
        with open(args.pbn, "w") as fh:
            for i, a in enumerate(auctions):
                v = {(0, 0): "None", (1, 0): "NS", (0, 1): "EW", (1, 1): "All"}[tuple(vul[i])]
                fh.write(f"{owners_to_pbn(owners[i])}\tdealer {SEATS[dealer[i]]}\tvul {v}\t"
                         f"{' '.join(call_name(c) for c in a)}\n")
        print("wrote", args.pbn)


if __name__ == "__main__":
    main()
