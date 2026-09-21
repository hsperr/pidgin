"""PyTorch port of the brl bridge bidding net (Kita et al., IEEE CoG 2024,
https://github.com/harukaki/brl) as a player for experiments/match/match.py.

Weights: run export_reference.py once in experiments/brl/.venv to turn the haiku .pkl into
``<name>_weights.npz``. check_port.py proves observation and logits match pgx 1.4.0 + JAX.

Observation (pgx.bridge_bidding._observe, 480 bools, actor-relative):
  [0:4]     vul [we not, we vul, they not, they vul]
  [4:8]     passes before the first bid, by relative seat (0 me, 1 LHO, 2 partner, 3 RHO)
  [8:428]   bid b (1C=0..7NT=34): 8 + 12b + rel made it; +4 doubled it; +8 redoubled it
  [428:480] own hand, OpenSpiel card index = suit(C0 D1 H2 S3) + rank(2=0..A=12) * 4
Actions: pgx 0 Pass, 1 X, 2 XX, 3 + k bid k. Ours: bid k = k, 35 Pass, 36 X, 37 XX.
Greedy argmax over legal calls, as brl's own WBridge5 client (distrax ``pi.mode()``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

L, PASS, DOUBLE, REDOUBLE = 35, 35, 36, 37

# our card c = suit*13 + rank, suits S H D C, ranks A..2  ->  OpenSpiel index
_OUR_TO_OS = torch.tensor([(3 - c // 13) + (12 - c % 13) * 4 for c in range(52)])
_PGX_TO_OURS = torch.cat((torch.tensor([PASS, DOUBLE, REDOUBLE]), torch.arange(L)))
_OURS_TO_PGX = torch.empty(38, dtype=torch.long)
_OURS_TO_PGX[_PGX_TO_OURS] = torch.arange(38)


class BrlNet(torch.nn.Module):
    """DeepMind variant: 4 x Linear(1024) + ReLU, then the 38-way actor head."""

    def __init__(self, weights: str | Path):
        super().__init__()
        w = np.load(weights)
        names = ["linear", "linear_1", "linear_2", "linear_3", "linear_4"]
        self.layers = torch.nn.ModuleList()
        for name in names:
            W = torch.from_numpy(w[f"actor_critic/{name}/w"]).float()   # haiku (in, out): y = x @ W + b
            lin = torch.nn.Linear(W.shape[0], W.shape[1])
            lin.weight.data.copy_(W.T)
            lin.bias.data.copy_(torch.from_numpy(w[f"actor_critic/{name}/b"]).float())
            self.layers.append(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers[:-1]:
            x = torch.relu(lin(x))
        return self.layers[-1](x)


def encode(hand: torch.Tensor, history: torch.Tensor, dealer: torch.Tensor,
           vul_ns: torch.Tensor, vul_ew: torch.Tensor, seat: torch.Tensor) -> torch.Tensor:
    """(B,480) pgx observation. hand (B,52) our card order; history (B,T) our call ids, -1 = none."""
    B, T = history.shape
    obs = torch.zeros(B, 480)
    rows = torch.arange(B)
    me_ns = seat % 2 == 0
    me_vul = torch.where(me_ns, vul_ns.bool(), vul_ew.bool())
    them_vul = torch.where(me_ns, vul_ew.bool(), vul_ns.bool())
    obs[:, 0], obs[:, 1] = (~me_vul).float(), me_vul.float()
    obs[:, 2], obs[:, 3] = (~them_vul).float(), them_vul.float()
    last = torch.full((B,), -1, dtype=torch.long)
    for t in range(T):
        a = history[:, t]
        rel = ((dealer + t) % 4 - seat) % 4
        opening_pass = (a == PASS) & (last < 0)
        obs[rows[opening_pass], 4 + rel[opening_pass]] = 1
        bid = (a >= 0) & (a < L)
        obs[rows[bid], 8 + 12 * a[bid] + rel[bid]] = 1
        last = torch.where(bid, a, last)
        for code, off in ((DOUBLE, 4), (REDOUBLE, 8)):
            m = (a == code) & (last >= 0)
            obs[rows[m], 8 + 12 * last[m] + off + rel[m]] = 1
    obs[:, 428 + _OUR_TO_OS] = hand.float()
    return obs


def legal_pgx(last: torch.Tensor, doubled: torch.Tensor, owner: torch.Tensor,
              seat: torch.Tensor) -> torch.Tensor:
    """(B,38) legal mask in pgx action order."""
    side = seat % 2
    m = torch.zeros(len(last), 38, dtype=torch.bool)
    m[:, 0] = True
    m[:, 1] = (last >= 0) & (doubled == 0) & (side != owner)
    m[:, 2] = (doubled == 1) & (side == owner)
    m[:, 3:] = torch.arange(L)[None] > last[:, None]
    return m


class BrlPlayer:
    """``brl:WEIGHTS.npz`` in match.py. Controls whichever seats the match gives it."""

    def __init__(self, weights: str):
        self.net = BrlNet(weights).eval()
        self.name = f"brl:{Path(weights).stem.removesuffix('_weights')}"
        self.meta = {"kind": "brl", "weights": str(weights),
                     "sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                     "source": "https://github.com/harukaki/brl", "rule": "greedy"}

    @torch.no_grad()
    def act(self, table, rows: torch.Tensor) -> torch.Tensor:
        seat = table.seat_abs(rows)
        hand = table.deals.hands[table.deal[rows], seat]
        obs = encode(hand, table.history[rows, :table.t], table.dealer[rows],
                     table.vul_ns[rows], table.vul_ew[rows], seat)
        logits = self.net(obs)
        legal = legal_pgx(table.st.last[rows], table.doubled[rows], table.contract_side[rows], seat)
        pick = logits.masked_fill(~legal, -torch.inf).argmax(-1)
        return _PGX_TO_OURS[pick]
