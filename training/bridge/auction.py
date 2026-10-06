"""Readable, immutable reference implementation of a bridge auction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .calls import CONTRACTS, DOUBLE, N_ACTIONS, PASS, REDOUBLE, format_call

SEAT_NAMES = ("N", "E", "S", "W")


@dataclass(frozen=True)
class AuctionState:
    dealer: int = 0
    vul_ns: bool = False
    vul_ew: bool = False
    calls: tuple[int, ...] = ()
    last_contract: int = -1
    contract_seat: int = -1
    doubled: int = 0  # 0=undoubled, 1=doubled, 2=redoubled
    pass_count: int = 0
    ended: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.dealer < 4:
            raise ValueError("dealer must be 0..3")

    @property
    def turn(self) -> int:
        return (self.dealer + len(self.calls)) % 4

    @property
    def side_to_act(self) -> int:
        return self.turn % 2

    @property
    def passed_out(self) -> bool:
        return self.ended and self.last_contract < 0

    def legal_mask(self) -> np.ndarray:
        legal = np.zeros(N_ACTIONS, dtype=np.bool_)
        if self.ended:
            return legal
        legal[self.last_contract + 1:PASS] = True
        legal[PASS] = True
        if self.last_contract >= 0:
            declaring_side = self.contract_seat % 2
            if self.doubled == 0 and self.side_to_act != declaring_side:
                legal[DOUBLE] = True
            elif self.doubled == 1 and self.side_to_act == declaring_side:
                legal[REDOUBLE] = True
        return legal

    def apply(self, action: int) -> "AuctionState":
        if self.ended:
            raise ValueError("cannot call after the auction has ended")
        if not 0 <= action < N_ACTIONS or not self.legal_mask()[action]:
            raise ValueError(
                f"illegal call {action} ({format_call(action) if 0 <= action < N_ACTIONS else '?'}) "
                f"by {SEAT_NAMES[self.turn]} after {self.format_history()}"
            )
        calls = self.calls + (action,)
        if action < PASS:
            return replace(self, calls=calls, last_contract=action,
                           contract_seat=self.turn, doubled=0, pass_count=0)
        if action == DOUBLE:
            return replace(self, calls=calls, doubled=1, pass_count=0)
        if action == REDOUBLE:
            return replace(self, calls=calls, doubled=2, pass_count=0)

        n_pass = self.pass_count + 1
        ended = (self.last_contract < 0 and n_pass >= 4) or (
            self.last_contract >= 0 and n_pass >= 3
        )
        return replace(self, calls=calls, pass_count=n_pass, ended=ended)

    def declarer(self) -> int:
        if self.last_contract < 0:
            return -1
        final_strain = CONTRACTS[self.last_contract][2]
        winning_side = self.contract_seat % 2
        for i, action in enumerate(self.calls):
            if action < PASS and action % 5 == final_strain:
                seat = (self.dealer + i) % 4
                if seat % 2 == winning_side:
                    return seat
        raise AssertionError("winning side never named final strain")

    def vulnerable(self, side: int) -> bool:
        return self.vul_ns if side == 0 else self.vul_ew

    def final_contract(self) -> str:
        if self.last_contract < 0:
            return "Passed out"
        suffix = "" if self.doubled == 0 else ("X" if self.doubled == 1 else "XX")
        return f"{format_call(self.last_contract)}{suffix} by {SEAT_NAMES[self.declarer()]}"

    def format_history(self) -> str:
        if not self.calls:
            return "(empty)"
        return " ".join(
            f"{SEAT_NAMES[(self.dealer + i) % 4]}:{format_call(a)}"
            for i, a in enumerate(self.calls)
        )

    @classmethod
    def from_calls(
        cls,
        calls: Iterable[int],
        dealer: int = 0,
        vul_ns: bool = False,
        vul_ew: bool = False,
    ) -> "AuctionState":
        state = cls(dealer=dealer, vul_ns=vul_ns, vul_ew=vul_ew)
        for call in calls:
            state = state.apply(int(call))
        return state

