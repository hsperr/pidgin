"""Let a model bid N deals against itself and save the auctions.

    python scripts/generate_auctions.py --n 1000 --out auctions.npz
    python scripts/generate_auctions.py --n 100 --pbn auctions.txt           # readable text
    python scripts/generate_auctions.py --n 1000000 --model PidginV1 --out v1_1M.npz
    # card-play training data: deals with their double-dummy tables, sampled calls
    python scripts/generate_auctions.py --n 1000000 --data data/dds_results_100M.npy \\
        --start 0 --temperature 1,3,5 --out data/play/auctions_1M.npz

All four seats are the same model. With the default --temperature 0 it bids greedily,
using the bidding network without bidding search. A positive temperature samples calls; with a
list, each deal draws one temperature from it. Dealer and vulnerability are random
(seeded).

Deals are random unless --data names a DDS dataset; then the deals come from it and
their double-dummy tables are saved too. The card-play trainers need those tables.

The .npz (the format the card-play trainers read):
  calls     (N, T) int8, padded with -1 (0..34 = 1C..7NT, 35 Pass, 36 X, 37 XX)
  hands     (N, 4, 52) uint8, seats N E S W; card = suit * 13 + rank, suits S H D C, A high
  dealer    (N,) int8;  vul_ns, vul_ew (N,) bool;  temp (N,) float32
  tricks    (N, 4, 5) int8 double-dummy tricks per seat and strain (with --data)
  deal_first  index of deal 0 in --data
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pidgin import SEATS, call_name, engine, load_bidder
from training.bridge.auction import AuctionState


def deals(args):
    """(owners (N, 52), tricks (N, 4, 5) or None, first index)."""
    if args.data:
        from training.bridge.deals import load_dataset
        owners, tricks = load_dataset(args.data)
        start = args.start if args.start >= 0 else len(owners) + args.start
        if start + args.n > len(owners):
            raise SystemExit(f"only {len(owners) - start} deals from {start}")
        sl = slice(start, start + args.n)
        return np.asarray(owners[sl], dtype=np.uint8), np.asarray(tricks[sl], dtype=np.int8), start
    rng = np.random.default_rng(args.seed)
    base = np.repeat(np.arange(4, dtype=np.uint8), 13)
    return np.stack([rng.permutation(base) for _ in range(args.n)]), None, 0


@torch.no_grad()
def bid_batch(bot, owners, dealer, vul, temp=None, gen=None):
    """Auctions for a batch of deals, one forward pass per round of calls.

    `temp` (B,): 0 = greedy (the server's rule), > 0 = sample from policy ** (1 / T)."""
    b = len(owners)
    hands = torch.as_tensor((owners[:, None, :] == np.arange(4)[None, :, None]).astype(np.float32))
    temp = torch.zeros(b) if temp is None else torch.as_tensor(temp, dtype=torch.float32)
    states = [AuctionState.from_calls([], dealer=int(d)) for d in dealer]
    history = [[] for _ in range(b)]
    batched = hasattr(bot, "batch_log_probs")
    while True:
        live = [i for i in range(b) if not states[i].ended]
        if not live:
            break
        seat = torch.tensor([(int(dealer[i]) + len(history[i])) % 4 for i in live])
        legal = torch.tensor([engine.legal_calls(bot, states[i]) for i in live])
        if batched:
            t = max(len(history[i]) for i in live)
            hist = torch.full((len(live), t), -1, dtype=torch.long)
            for r, i in enumerate(live):
                hist[r, :len(history[i])] = torch.tensor(history[i], dtype=torch.long)
            lp = bot.batch_log_probs(hands[live, seat], hist, torch.as_tensor(dealer[live]).long(),
                                     torch.as_tensor(vul[live, 0]).bool(), torch.as_tensor(vul[live, 1]).bool(),
                                     seat, legal)
        else:  # a bot without a batched path: one position at a time
            lp = torch.stack([torch.tensor(bot.decide(hands[i, s][None], history[i], int(dealer[i]),
                                                      (bool(vul[i, 0]), bool(vul[i, 1])),
                                                      legal[r].tolist())["policy"]).log()
                              for r, (i, s) in enumerate(zip(live, seat.tolist()))])
        picks = lp.argmax(1)
        t_live = temp[live]
        hot = t_live > 0
        if hot.any():
            scaled = (lp[hot] / t_live[hot, None]).masked_fill(~legal[hot], -torch.inf)
            picks[hot] = torch.multinomial(scaled.softmax(1), 1, generator=gen).squeeze(1)
        for i, c in zip(live, picks.tolist()):
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
    ap.add_argument("--temperature", default="0", help="0 = greedy; e.g. 1,3,5 = one drawn per deal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    bot = load_bidder(args.model)
    owners, tricks, first = deals(args)
    n = len(owners)
    gen = torch.Generator().manual_seed(args.seed)
    dealer = torch.randint(4, (n,), generator=gen).numpy()
    vul = torch.randint(2, (n, 2), generator=gen).numpy().astype(np.uint8)
    temps = torch.tensor([float(t) for t in args.temperature.split(",")])
    temp = temps[torch.randint(len(temps), (n,), generator=gen)].numpy()

    auctions, t0 = [], time.time()
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        auctions += bid_batch(bot, owners[s:e], dealer[s:e], vul[s:e], temp[s:e], gen)
        print(f"{e}/{n} auctions, {time.time() - t0:.0f}s", flush=True)

    calls = np.full((n, max(map(len, auctions))), -1, dtype=np.int8)
    for i, a in enumerate(auctions):
        calls[i, :len(a)] = a
    hands = (owners[:, None, :] == np.arange(4)[None, :, None]).astype(np.uint8)
    out = dict(calls=calls, hands=hands, dealer=dealer.astype(np.int8), vul_ns=vul[:, 0].astype(bool),
               vul_ew=vul[:, 1].astype(bool), temp=temp.astype(np.float32), deal_first=first,
               model=np.array(args.model))
    if tricks is not None:
        out["tricks"] = tricks
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, calls.shape)

    if args.pbn:
        from emergent.deck import owners_to_pbn
        names = {(0, 0): "None", (1, 0): "NS", (0, 1): "EW", (1, 1): "All"}
        with open(args.pbn, "w") as fh:
            for i, a in enumerate(auctions):
                fh.write(f"{owners_to_pbn(owners[i])}\tdealer {SEATS[dealer[i]]}\t"
                         f"vul {names[tuple(vul[i])]}\t{' '.join(call_name(c) for c in a)}\n")
        print("wrote", args.pbn)


if __name__ == "__main__":
    main()
