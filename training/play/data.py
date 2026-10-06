"""Turn generated auctions into contracts the play engine can start from.

Input is whatever ``experiments/play/gen_auctions.py`` wrote. Passed-out deals are
dropped: there is nothing to play.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.bridge.calls import STRAIN_PERM, DOUBLE, PASS, REDOUBLE

BID_STRAIN_TO_TRUMP = STRAIN_PERM         # bids are C,D,H,S,NT; cards and DDS are S,H,D,C,NT
N_CALLS = 38


@dataclass
class Contracts:
    """One row per playable deal, in the file's order."""

    owner: torch.Tensor       # (n, 52) long, seat holding each card
    trump: torch.Tensor       # (n,) long, 0..3 suit or 4 notrump
    declarer: torch.Tensor    # (n,) long
    level: torch.Tensor       # (n,) long, 1..7
    doubled: torch.Tensor     # (n,) long, 0 none, 1 doubled, 2 redoubled
    vul_ns: torch.Tensor      # (n,) bool
    vul_ew: torch.Tensor      # (n,) bool
    call_bag: torch.Tensor    # (n, 4, 38) float, which calls each seat made
    calls: torch.Tensor       # (n, max) long, the auction in order, 38 = padding
    n_calls: torch.Tensor     # (n,) long, how many calls the auction really had
    dealer: torch.Tensor      # (n,) long, who called first
    dd_tricks: torch.Tensor   # (n,) long, double dummy tricks for this declarer and strain

    def __len__(self) -> int:
        return len(self.owner)

    def to(self, device) -> "Contracts":
        moved = {k: v.to(device) for k, v in self.__dict__.items()}
        return Contracts(**moved)

    def subset(self, index) -> "Contracts":
        return Contracts(**{k: v[index] for k, v in self.__dict__.items()})


def load_contracts(path: str | Path, limit: int | None = None) -> Contracts:
    raw = np.load(Path(path))
    stop = len(raw["calls"]) if limit is None else min(limit, len(raw["calls"]))
    calls, dealer = raw["calls"][:stop], raw["dealer"][:stop].astype(np.int64)
    hands, tricks = raw["hands"][:stop], raw["tricks"][:stop].astype(np.int64)
    vul_ns, vul_ew = raw["vul_ns"][:stop], raw["vul_ew"][:stop]

    n = stop
    keep, trump, declarer, level, doubled, dd = [], [], [], [], [], []
    width = calls.shape[1]
    seqs = np.full((n, width), N_CALLS, dtype=np.int64)     # N_CALLS is the padding token
    lengths = np.zeros(n, dtype=np.int64)
    bag = np.zeros((n, 4, N_CALLS), dtype=np.float32)
    for row in range(n):
        seq = [int(c) for c in calls[row] if c >= 0]
        seqs[row, :len(seq)] = seq
        lengths[row] = len(seq)
        for i, call in enumerate(seq):
            bag[row, (dealer[row] + i) % 4, call] = 1.0
        bids = [(i, c) for i, c in enumerate(seq) if c < PASS]
        if not bids:
            continue                                            # passed out
        last_i, last = bids[-1]
        bid_strain = last % 5
        side = (dealer[row] + last_i) % 4 % 2
        first = next(i for i, c in bids
                     if c % 5 == bid_strain and (dealer[row] + i) % 4 % 2 == side)
        seat = (dealer[row] + first) % 4
        tail = seq[last_i + 1:]
        keep.append(row)
        trump.append(BID_STRAIN_TO_TRUMP[bid_strain])
        declarer.append(seat)
        level.append(last // 5 + 1)
        doubled.append(2 if REDOUBLE in tail else (1 if DOUBLE in tail else 0))
        dd.append(int(tricks[row, seat, BID_STRAIN_TO_TRUMP[bid_strain]]))

    keep = np.asarray(keep)
    owner = torch.from_numpy(hands[keep].argmax(1).astype(np.int64))
    return Contracts(
        owner=owner,
        trump=torch.tensor(trump, dtype=torch.long),
        declarer=torch.tensor(declarer, dtype=torch.long),
        level=torch.tensor(level, dtype=torch.long),
        doubled=torch.tensor(doubled, dtype=torch.long),
        vul_ns=torch.from_numpy(vul_ns[keep].astype(bool)),
        vul_ew=torch.from_numpy(vul_ew[keep].astype(bool)),
        call_bag=torch.from_numpy(bag[keep]),
        calls=torch.from_numpy(seqs[keep]),
        n_calls=torch.from_numpy(lengths[keep]),
        dealer=torch.from_numpy(dealer[keep]),
        dd_tricks=torch.tensor(dd, dtype=torch.long),
    )
