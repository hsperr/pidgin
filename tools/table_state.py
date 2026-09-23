"""Batched auction bookkeeping for the duplicate match table.

A small, self-contained state: which rungs each side has bid, the standing
contract, who bid it first, and the pass count. ``tools/match.py`` drives it;
nothing here trains, and nothing here sees a card.

Rungs are the 35 contracts, low to high. Pass is action index ``L``; the two
slots above it are Double and Redouble, which the table applies itself.
"""

from __future__ import annotations

import numpy as np
import torch

STRAINS = ["C", "D", "H", "S", "NT"]          # trick-value ordering
CONTRACTS = [(f"{lvl}{st}", lvl, si)
             for lvl in range(1, 8)
             for si, st in enumerate(STRAINS)]


def contract_strains() -> np.ndarray:
    """Strain index of each rung."""
    return np.array([si for _, _, si in CONTRACTS], dtype=np.int64)


class St:
    """A batch of auction positions, in ABSOLUTE seat coordinates.

    ``deal`` says which deal each row belongs to, so hands and double-dummy
    tricks are looked up rather than carried around. ``dbl``/``dblflag`` are
    the doubling state; this table keeps its own, so they stay zero here and
    exist only to keep the state shape stable.
    """

    __slots__ = ("bid", "dbl", "last", "lastseat", "dblflag", "npass",
                 "alive", "deal")

    def __init__(self, bid, dbl, last, lastseat, dblflag, npass, alive, deal):
        self.bid, self.dbl = bid, dbl
        self.last, self.lastseat = last, lastseat
        self.dblflag, self.npass = dblflag, npass
        self.alive, self.deal = alive, deal

    @staticmethod
    def empty(deal, L):
        n, dev = deal.shape[0], deal.device
        z = lambda: torch.zeros(n, dtype=torch.long, device=dev)   # noqa: E731
        return St(torch.zeros(n, L, 4, device=dev),
                  torch.zeros(n, L, 4, device=dev),
                  torch.full((n,), -1, dtype=torch.long, device=dev),
                  torch.full((n,), -1, dtype=torch.long, device=dev),
                  z(), z(),
                  torch.ones(n, dtype=torch.bool, device=dev), deal)

    def repeat(self, k):
        r = lambda x: x.repeat_interleave(k, 0)                    # noqa: E731
        return St(r(self.bid), r(self.dbl), r(self.last), r(self.lastseat),
                  r(self.dblflag), r(self.npass), r(self.alive), r(self.deal))

    def __getitem__(self, sl):
        return St(self.bid[sl], self.dbl[sl], self.last[sl], self.lastseat[sl],
                  self.dblflag[sl], self.npass[sl], self.alive[sl],
                  self.deal[sl])

    @property
    def n(self):
        return self.last.shape[0]


def apply_call_(st, call, seat, L):
    """One turn, written INTO the state instead of copying it.

    The two history grids are (N, 35, 4) floats, so cloning them every turn of
    every auction is gigabytes of memcpy. Only safe where the caller owns the
    state exclusively. A dead row is a no-op: every write is gated on ``alive``.
    """
    n, dev = st.n, st.last.device
    rows = torch.arange(n, device=dev)
    act = st.alive
    is_bid = act & (call < L)
    is_pass = act & (call == L)
    ci = call.clamp(max=L - 1)
    st.bid[rows, ci, seat] = torch.where(is_bid, torch.ones(n, device=dev),
                                         st.bid[rows, ci, seat])
    st.last = torch.where(is_bid, ci, st.last)
    st.lastseat = torch.where(is_bid, torch.full_like(st.lastseat, seat),
                              st.lastseat)
    st.npass = torch.where(is_pass, st.npass + 1,
                           torch.where(act, torch.zeros_like(st.npass),
                                       st.npass))
    over = is_pass & (((st.npass >= 3) & (st.last >= 0)) | (st.npass >= 4))
    st.alive = st.alive & ~over
    return st


def declarer_of(st, strain_si, L):
    """(N,) the seat that first named the final strain for the winning side.

    Rows with no contract return seat 0; their score is forced to 0 anyway.
    """
    n, dev = st.n, st.last.device
    c = st.last.clamp(min=0)
    side = (st.lastseat.clamp(min=0) % 2)
    s1, s2 = side, side + 2
    g = lambda s: st.bid.gather(2, s.view(-1, 1, 1).expand(n, L, 1)).squeeze(2)  # noqa: E731
    b1, b2 = g(s1), g(s2)
    same = strain_si[None, :] == strain_si[c][:, None]
    cand = same & ((b1 + b2) > 0)
    rungs = torch.arange(L, device=dev)[None, :].expand(n, L)
    k0 = torch.where(cand, rungs, torch.full_like(rungs, L)).min(1).values
    k0 = k0.clamp(max=L - 1)
    mine = b1.gather(1, k0[:, None]).squeeze(1) > 0
    return torch.where(mine, s1, s2)
