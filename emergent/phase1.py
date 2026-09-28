"""The phase 1 bidding net (`SeatNet`), as the bid desk runs it: inference only.

Cut out of the phase 1 experiment scripts (exp11four.py, exp10four.py and
fullinfo.py in ~/code/bridge/emergent), which also carried the trainer, the
rollouts and the scoring. The classes keep their parameter names, so the old
checkpoints (`models/own_s16_step5250.pt`) load unchanged. Calls are indexed as
everywhere else here: bid k = k (1C = 0 ... 7NT = 34), Pass = L, Double = L + 1.
Dealer is North and nobody is vulnerable: the net has no input for either.
"""
import torch
import torch.nn as nn

L = 35                       # contracts


class St:
    """A batch of auction positions, in absolute seat coordinates."""

    __slots__ = ("bid", "dbl", "last", "lastseat", "dblflag", "npass", "alive")

    def __init__(self, bid, dbl, last, lastseat, dblflag, npass, alive):
        self.bid, self.dbl = bid, dbl
        self.last, self.lastseat = last, lastseat
        self.dblflag, self.npass = dblflag, npass
        self.alive = alive

    @staticmethod
    def empty(n, L=L):
        z = lambda: torch.zeros(n, dtype=torch.long)
        return St(torch.zeros(n, L, 4), torch.zeros(n, L, 4),
                  torch.full((n,), -1, dtype=torch.long), torch.full((n,), -1, dtype=torch.long),
                  z(), z(), torch.ones(n, dtype=torch.bool))

    @property
    def n(self):
        return self.last.shape[0]


def apply_call_(st, call, seat, L=L):
    """Apply `call` (a (n,) tensor) by `seat` to `st`, in place."""
    n = st.n
    rows = torch.arange(n)
    act = st.alive
    is_bid = act & (call < L)
    is_pass = act & (call == L)
    is_dbl = act & (call == L + 1)
    ci = call.clamp(max=L - 1)
    lc = st.last.clamp(min=0)
    st.bid[rows, ci, seat] = torch.where(is_bid, torch.ones(n), st.bid[rows, ci, seat])
    st.dbl[rows, lc, seat] = torch.where(is_dbl, torch.ones(n), st.dbl[rows, lc, seat])
    st.last = torch.where(is_bid, ci, st.last)
    st.lastseat = torch.where(is_bid, torch.full_like(st.lastseat, seat), st.lastseat)
    st.dblflag = torch.where(is_bid, torch.zeros_like(st.dblflag),
                             torch.where(is_dbl, torch.ones_like(st.dblflag), st.dblflag))
    st.npass = torch.where(is_pass, st.npass + 1,
                           torch.where(act, torch.zeros_like(st.npass), st.npass))
    over = is_pass & (((st.npass >= 3) & (st.last >= 0)) | (st.npass >= 4))
    st.alive = st.alive & ~over
    return st


class SuitEncoder(nn.Module):
    """One hand -> a vector: the same small net reads each suit, outputs kept in ladder order."""

    def __init__(self, d=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(13, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.out_dim = 4 * d

    def forward(self, hand):
        B = hand.shape[0]
        return self.f(hand.view(B * 4, 13)).view(B, -1)


class SeatNet(nn.Module):
    """(my hand, the auction rotated to me) -> my side's score for each call.

    No seat identity: the only thing that tells one chair from another is which
    slot of the rotated history is mine, plus the turn index.
    """

    def __init__(self, L=L, hidden=512, d_hand=64, d_rung=48, layers=3, dbl_head=False):
        super().__init__()
        self.L, self.d = L, d_rung
        self.hand_enc = SuitEncoder(d_hand)
        self.rung = nn.Embedding(L, d_rung)
        self.no_bid = nn.Parameter(torch.zeros(d_rung))
        self.pass_key = nn.Parameter(torch.zeros(d_rung))
        self.dbl_key = nn.Parameter(torch.zeros(d_rung))
        d_in = self.hand_enc.out_dim + 9 * d_rung + 6
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.proj = nn.Linear(d, d_rung)
        self.bias = nn.Parameter(torch.zeros(L + 2))
        self.dbl_head = None
        if dbl_head:              # Double read through its own two layers (ticket 006)
            self.dbl_head = nn.Sequential(nn.Linear(d, d_rung), nn.ReLU(), nn.Linear(d_rung, 1))

    def _rung_map(self, seat):
        """(4L, 4d): the history's rotation to `seat` folded into the rung table."""
        W = self.rung.weight
        big = W.new_zeros(self.L * 4, 4 * self.d)
        for j in range(4):
            big[(seat + j) % 4::4, j * self.d:(j + 1) * self.d] = W
        return big

    def _sums(self, h, seat, rmap=None):
        n = h.shape[0]
        if rmap is None:
            rmap = self._rung_map(seat)
        return h.reshape(n, self.L * 4) @ rmap

    def _trunk(self, hand, st, seat, t, hv=None):
        """`hv` is a precomputed `hand_enc(hand)`; then `hand` is not read."""
        n = st.n
        if hv is None:
            hv = self.hand_enc(hand)
        lastv = torch.where(st.last[:, None] >= 0, self.rung(st.last.clamp(min=0)),
                            self.no_bid[None].expand(n, self.d))
        mine = ((st.lastseat >= 0) & ((st.lastseat % 2) == (seat % 2))).float()
        scal = torch.stack([torch.full((n,), t / 10.0), st.last.float() / self.L,
                            st.npass.float() / 3.0, st.dblflag.float(), mine,
                            (st.last >= 0).float()], -1)
        rmap = self._rung_map(seat)
        return self.body(torch.cat([hv, self._sums(st.bid, seat, rmap),
                                    self._sums(st.dbl, seat, rmap), lastv, scal], -1))

    def _base_q(self, h):
        """(N, L+2): one Q per call."""
        if self.dbl_head is None:
            keys = torch.cat([self.rung.weight, self.pass_key[None], self.dbl_key[None]], 0)
            return self.proj(h) @ keys.T + self.bias
        keys = torch.cat([self.rung.weight, self.pass_key[None]], 0)
        q = self.proj(h) @ keys.T + self.bias[:self.L + 1]
        return torch.cat([q, self.dbl_head(h)], -1)

    def forward(self, hand, st, seat, t, hv=None):
        return self._base_q(self._trunk(hand, st, seat, t, hv))
