"""Vectorized four-seat auctions, opponent-aware features, and own-bid scores.

Rules: 35 contracts + Pass (``FourSeatBatch``), optionally + Double
(``FourSeatDoubleBatch``; Redouble is never legal; ``competitive.FourSeatRedoubleBatch``
adds it). Ascending legality; the
auction ends after three Passes following a bid or Double, or four Passes at the
start. A row may have a ``silent`` side (0=NS, 1=EW) that is forced to Pass:
that is the existing cooperative game.

Own-bid reward (phase-1 style, deliberately not zero-sum): each side is scored
on its OWN highest bid as an undoubled contract with its own vulnerability,
declared by the first player of that side to name that strain; 0 if the side
never bid. Doubles never change this number. Training units are
``(score - side cooperative DD ceiling) / 100``.

Feature layout, actor-relative:

- ``[0:77]`` the cooperative partnership view of ``contract/environment.py``
  (own bids, partner bids, self/partner passed before any own-partnership bid,
  dealer relative to actor, own-side vulnerable), same semantics;
- ``[77:112]`` bids by LHO (actor+1), ``[112:147]`` bids by RHO (actor+3);
- double models only: ``[147]`` standing contract is doubled, ``[148]`` it was
  doubled by the actor's side.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..bridge.calls import CONTRACTS, DOUBLE, PASS
from ..bridge.scoring import contract_score
from ..contract.data import TorchDeals
from ..contract.environment import AUCTION_FEATURES, PARTNER_PASSED, SELF_PASSED
from ..contract.targets import CONTRACT_TABLE_STRAIN, SCORE_LOOKUP, TorchScorer

OPPONENT_FEATURES = 70
FOURSEAT_FEATURES = AUCTION_FEATURES + OPPONENT_FEATURES
LHO_BIDS = slice(AUCTION_FEATURES, AUCTION_FEATURES + 35)
RHO_BIDS = slice(AUCTION_FEATURES + 35, FOURSEAT_FEATURES)
DOUBLE_FEATURES = 2
STANDING_DOUBLED = FOURSEAT_FEATURES
DOUBLED_BY_MY_SIDE = FOURSEAT_FEATURES + 1
FOURSEAT_DOUBLE_FEATURES = FOURSEAT_FEATURES + DOUBLE_FEATURES
# The longest legal auction with Pass/Double: 3 opening passes, then per bid
# at most "X P P" or "P P X P P" (5 calls) before the next bid, 3 final passes
# after the last one = 3 + 35 + 34*5 + 5 = 213.
MAX_CALLS = 216
# Bid-bit block start by seat relative to the actor: self, LHO, partner, RHO.
_BLOCK_START = (0, LHO_BIDS.start, 35, RHO_BIDS.start)
_TABLE_STRAIN = torch.as_tensor(CONTRACT_TABLE_STRAIN)
_UNDOUBLED = torch.as_tensor(SCORE_LOOKUP, dtype=torch.float32)            # [vul, c, tricks]
_DOUBLED = torch.tensor([[[contract_score(level, strain, k, 1, bool(v)) for k in range(14)]
                          for _, level, strain in CONTRACTS] for v in (0, 1)],
                        dtype=torch.float32)
_REDOUBLED = torch.tensor([[[contract_score(level, strain, k, 2, bool(v)) for k in range(14)]
                            for _, level, strain in CONTRACTS] for v in (0, 1)],
                          dtype=torch.float32)


def _last_positions(history: torch.Tensor):
    """Position of the last bid and of the last Double per row (-1 if none)."""
    pos = torch.arange(history.shape[1], device=history.device)[None]
    bid = (history >= 0) & (history < PASS)
    none = torch.full_like(history, -1)
    last_bid = torch.where(bid, pos, none).max(1).values
    last_x = torch.where(history == DOUBLE, pos, none).max(1).values
    return last_bid, last_x


def features_from_history(history: torch.Tensor, dealer: torch.Tensor, vul_ns: torch.Tensor,
                          vul_ew: torch.Tensor, actor: torch.Tensor,
                          doubles: bool = False) -> torch.Tensor:
    """Features ``(B,147)`` (``(B,149)`` with ``doubles``) from call history ``(B,T)``.

    Call ``i`` of a row was made by seat ``(dealer + i) % 4``; -1 is padding.
    Double calls never set bid or passed-first bits.
    """
    n, length = history.shape
    device = history.device
    width = FOURSEAT_DOUBLE_FEATURES if doubles else FOURSEAT_FEATURES
    feats = torch.zeros(n, width, device=device)
    if length:
        pos = torch.arange(length, device=device)
        rel = ((dealer[:, None] + pos[None]) - actor[:, None]) % 4
        valid = history >= 0
        bid = valid & (history < PASS)
        block = torch.as_tensor(_BLOCK_START, device=device)[rel]
        index = torch.where(bid, block + history, torch.zeros_like(history))
        feats.scatter_add_(1, index, bid.float())
        own_side = rel % 2 == 0
        own_bid = (bid & own_side).long()
        seen = (own_bid.cumsum(1) - own_bid) > 0
        early_pass = valid & (history == PASS) & own_side & ~seen
        feats[:, SELF_PASSED] = (early_pass & (rel == 0)).any(1).float()
        feats[:, PARTNER_PASSED] = (early_pass & (rel == 2)).any(1).float()
        if doubles:
            last_bid, last_x = _last_positions(history)
            standing = last_x > last_bid
            x_rel = rel.gather(1, last_x.clamp(min=0)[:, None]).squeeze(1)
            feats[:, STANDING_DOUBLED] = standing.float()
            feats[:, DOUBLED_BY_MY_SIDE] = (standing & (x_rel % 2 == 0)).float()
    rows = torch.arange(n, device=device)
    feats[rows, 72 + (dealer - actor) % 4] = 1.0
    feats[:, 76] = torch.where(actor % 2 == 0, vul_ns, vul_ew).float()
    return feats


@dataclass
class FourSeatBatch:
    deal: torch.Tensor        # (B,) index into TorchDeals
    dealer: torch.Tensor      # (B,) absolute seat
    vul: torch.Tensor         # (B,2) bool: NS, EW vulnerable
    silent: torch.Tensor      # (B,) side forced to Pass, -1 for none
    history: torch.Tensor     # (B,MAX_CALLS) calls, -1 padding
    t: torch.Tensor           # calls made
    last: torch.Tensor        # standing contract or -1
    pass_count: torch.Tensor  # consecutive passes
    ended: torch.Tensor

    n_actions = PASS + 1
    doubles = False

    @classmethod
    def start(cls, deal, dealer, vul_ns, vul_ew, silent=None) -> "FourSeatBatch":
        b = len(deal)
        dev = deal.device
        zeros = torch.zeros(b, dtype=torch.long, device=dev)
        if silent is None:
            silent = torch.full((b,), -1, dtype=torch.long, device=dev)
        return FourSeatBatch(
            deal, dealer.long(), torch.stack((vul_ns.bool(), vul_ew.bool()), 1),
            silent.long(), torch.full((b, MAX_CALLS), -1, dtype=torch.long, device=dev),
            zeros.clone(), torch.full((b,), -1, dtype=torch.long, device=dev),
            zeros.clone(), torch.zeros(b, dtype=torch.bool, device=dev))

    @classmethod
    def cat(cls, batches: list["FourSeatBatch"]) -> "FourSeatBatch":
        return cls(**{f.name: torch.cat([getattr(b, f.name) for b in batches])
                      for f in fields(cls)})

    def __len__(self) -> int:
        return len(self.deal)

    def subset(self, rows: torch.Tensor) -> "FourSeatBatch":
        return type(self)(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @property
    def actor_seat(self) -> torch.Tensor:
        return (self.dealer + self.t) % 4

    @property
    def side(self) -> torch.Tensor:
        return self.actor_seat % 2

    @property
    def forced(self) -> torch.Tensor:
        """Live rows whose player to act belongs to the silent side."""
        return ~self.ended & (self.side == self.silent)

    def legal(self) -> torch.Tensor:
        """``(B,36)``: contracts above the standing one (from any seat), and Pass."""
        contracts = torch.arange(35, device=self.last.device)[None] > self.last[:, None]
        contracts &= ~self.forced[:, None]
        mask = torch.cat([contracts, torch.ones_like(contracts[:, :1])], dim=1)
        return mask & ~self.ended[:, None]

    def features(self) -> torch.Tensor:
        length = int(self.t.max()) if len(self) else 0
        return features_from_history(self.history[:, :length], self.dealer,
                                     self.vul[:, 0], self.vul[:, 1], self.actor_seat,
                                     self.doubles)

    def apply(self, action: torch.Tensor, rows: torch.Tensor) -> None:
        """In-place: rows (bool mask) make ``action[rows]``."""
        r = rows.nonzero().squeeze(1)
        if len(r) == 0:
            return
        a = action[r]
        if (bool(((a < 0) | (a >= self.n_actions)).any())
                or not bool(self.legal()[r, a].all())):
            raise ValueError("illegal four-seat call in batch")
        if bool((self.t[r] >= self.history.shape[1]).any()):
            raise AssertionError("four-seat auction exceeded MAX_CALLS")
        seat = self.actor_seat[r]
        self.history[r, self.t[r]] = a
        bid = a < PASS
        self.last[r[bid]] = a[bid]
        self.pass_count[r[bid]] = 0
        self._after_call(r, a, seat)
        rp = r[a == PASS]
        self.pass_count[rp] += 1
        self.ended[rp] = (((self.last[rp] < 0) & (self.pass_count[rp] >= 4))
                          | ((self.last[rp] >= 0) & (self.pass_count[rp] >= 3)))
        self.t[r] += 1

    def _after_call(self, r, a, seat) -> None:
        """Hook for subclasses to update extra state before the pass count."""

    def bid_seats(self) -> torch.Tensor:
        """``(B,35)`` absolute seat that bid each contract, -1 if nobody."""
        out = torch.full((len(self), 35), -1, dtype=torch.long, device=self.deal.device)
        rows, pos = ((self.history >= 0) & (self.history < PASS)).nonzero(as_tuple=True)
        out[rows, self.history[rows, pos]] = (self.dealer[rows] + pos) % 4
        return out

    def call_lists(self) -> list[list[int]]:
        return [[int(x) for x in row if x >= 0] for row in self.history.cpu()]


@dataclass
class FourSeatDoubleBatch(FourSeatBatch):
    """Four-seat auction with Double (action 36). Redouble is never legal."""

    contract_seat: torch.Tensor = None   # seat of the standing bid, -1 if none
    doubled: torch.Tensor = None         # standing contract doubled

    n_actions = DOUBLE + 1
    doubles = True

    @classmethod
    def start(cls, deal, dealer, vul_ns, vul_ew, silent=None) -> "FourSeatDoubleBatch":
        base = FourSeatBatch.start(deal, dealer, vul_ns, vul_ew, silent)
        return cls(**{f.name: getattr(base, f.name) for f in fields(FourSeatBatch)},
                   contract_seat=torch.full_like(base.last, -1),
                   doubled=torch.zeros_like(base.ended))

    def can_double(self) -> torch.Tensor:
        return (~self.ended & ~self.forced & (self.last >= 0) & ~self.doubled
                & (self.contract_seat % 2 != self.side))

    def legal(self) -> torch.Tensor:
        return torch.cat([super().legal(), self.can_double()[:, None]], 1)

    def _after_call(self, r, a, seat) -> None:
        bid = a < PASS
        self.contract_seat[r[bid]] = seat[bid]
        self.doubled[r[bid]] = False
        x = r[a == DOUBLE]
        self.doubled[x] = True
        self.pass_count[x] = 0

    def standing_double_position(self) -> torch.Tensor:
        """History position of the Double on the standing contract, -1 if undoubled."""
        last_bid, last_x = _last_positions(self.history)
        return torch.where(last_x > last_bid, last_x, torch.full_like(last_x, -1))


def contract_declarer(batch: FourSeatBatch) -> torch.Tensor:
    """Declarer seat of the standing contract, -1 if there is none."""
    rows = torch.arange(len(batch), device=batch.deal.device)
    ladder = torch.arange(35, device=batch.deal.device)[None]
    seats = batch.bid_seats()
    last = batch.last.clamp(min=0)
    owner = seats[rows, last] % 2
    same = (seats >= 0) & (seats % 2 == owner[:, None]) & (ladder % 5 == (last % 5)[:, None])
    first = torch.where(same, ladder, torch.full_like(seats, 35)).min(1).values.clamp(max=34)
    return torch.where(batch.last >= 0, seats[rows, first], torch.full_like(last, -1))


def contract_results(batch: FourSeatBatch, deals: TorchDeals):
    """Declarer seat, undoubled and doubled declarer scores of the standing contract."""
    device = batch.deal.device
    rows = torch.arange(len(batch), device=device)
    declarer = contract_declarer(batch)
    c = batch.last.clamp(min=0)
    tricks = deals.tricks[batch.deal, declarer.clamp(min=0), _TABLE_STRAIN.to(device)[c]].long()
    vul = batch.vul[rows, declarer.clamp(min=0) % 2].long()
    return declarer, _UNDOUBLED.to(device)[vul, c, tricks], _DOUBLED.to(device)[vul, c, tricks]


def double_delta(batch: FourSeatBatch, deals: TorchDeals) -> torch.Tensor:
    """Defenders' (doubled - undoubled) result of the standing contract / 100; 0 if none."""
    _, undoubled, doubled = contract_results(batch, deals)
    delta = (undoubled - doubled) / 100.0
    return torch.where(batch.last >= 0, delta, torch.zeros_like(delta))


def table_ns_score(batch: FourSeatBatch, deals: TorchDeals) -> torch.Tensor:
    """Real NS table score, doubled (redoubled) when the final contract is doubled (redoubled)."""
    declarer, undoubled, doubled = contract_results(batch, deals)
    is_doubled = getattr(batch, "doubled", None)
    raw = undoubled if is_doubled is None else torch.where(is_doubled, doubled, undoubled)
    is_redoubled = getattr(batch, "redoubled", None)
    if is_redoubled is not None:
        c = batch.last.clamp(min=0)
        rows = torch.arange(len(batch), device=c.device)
        vul = batch.vul[rows, declarer.clamp(min=0) % 2].long()
        tricks = deals.tricks[batch.deal, declarer.clamp(min=0), _TABLE_STRAIN.to(c.device)[c]].long()
        raw = torch.where(is_redoubled, _REDOUBLED.to(c.device)[vul, c, tricks], raw)
    ns = torch.where(declarer % 2 == 0, raw, -raw)
    return torch.where(batch.last >= 0, ns, torch.zeros_like(ns))


@torch.no_grad()
def own_bid_scores(batch: FourSeatBatch, deals: TorchDeals, scorer: TorchScorer,
                   exclude: torch.Tensor | None = None):
    """Own-bid points ``(B,2)``, side DD ceilings ``(B,2)``, and undoubled table NS score.

    The side that owns the final contract has its own-bid score equal to the
    undoubled table result, so that score is derived from the same numbers.
    ``exclude`` ``(B,35)`` bool: bids ignored for the own-bid score (sacrifice bids);
    the table score still uses every bid.
    """
    n = len(batch)
    device = batch.deal.device
    rows = torch.arange(n, device=device)
    ladder = torch.arange(35, device=device)
    all_seats = batch.bid_seats()
    seats = all_seats if exclude is None else all_seats.masked_fill(exclude, -1)
    scores, ceilings = [], []
    for side in (0, 1):
        owned = (seats >= 0) & (seats % 2 == side)
        has = owned.any(1)
        top = torch.where(owned, ladder[None], torch.full_like(seats, -1)).max(1).values
        strain = top.clamp(min=0) % 5
        same = owned & (ladder[None] % 5 == strain[:, None])
        first = torch.where(same, ladder[None], torch.full_like(seats, 35)).min(1).values
        declarer = seats[rows, first.clamp(max=34)]
        rel = ((declarer - side) % 4) // 2
        exact = scorer.exact(deals.rel_tricks(batch.deal, torch.full_like(batch.deal, side)),
                             batch.vul[:, side])
        score = exact[rows, rel.clamp(min=0), top.clamp(min=0)]
        scores.append(torch.where(has, score, torch.zeros_like(score)))
        ceilings.append(scorer.ceiling(exact))
    score = torch.stack(scores, 1)
    ceiling = torch.stack(ceilings, 1)
    last = batch.last.clamp(min=0)
    ns_owns = all_seats[rows, last] % 2 == 0
    table_ns = torch.where(ns_owns, score[:, 0], -score[:, 1])
    table_ns = torch.where(batch.last >= 0, table_ns, torch.zeros_like(table_ns))
    return score, ceiling, table_ns
