"""Silent-opponent cooperative auction.

Reference implementation on top of the trusted four-seat ``AuctionState``:

- exactly one active partnership bids; the two inactive seats are forced to Pass;
- Double and Redouble are never legal;
- normal ascending legality and termination still apply;
- the partnership receives the undoubled score of the final contract, zero if
  passed out.

The vectorized training code in ``prefixes.py`` is tested against this class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.auction import AuctionState
from ..bridge.calls import DOUBLE, N_ACTIONS, PASS, REDOUBLE
from ..bridge.scoring import terminal_ns_score

N_COOP_ACTIONS = PASS + 1          # 35 contracts + Pass; X/XX have no slot
AUCTION_FEATURES = 35 * 2 + 2 + 4 + 1
# Actor-relative auction feature layout.
OWN_BIDS = slice(0, 35)
PARTNER_BIDS = slice(35, 70)
SELF_PASSED = 70          # actor passed before any partnership bid
PARTNER_PASSED = 71       # partner passed before any partnership bid
DEALER_REL = slice(72, 76)
OWN_VUL = 76


@dataclass(frozen=True)
class CooperativeAuction:
    state: AuctionState
    active_side: int

    @classmethod
    def new(cls, dealer: int, active_side: int, vulnerable: bool) -> "CooperativeAuction":
        if active_side not in (0, 1):
            raise ValueError("active_side must be 0 (NS) or 1 (EW)")
        state = AuctionState(dealer=dealer,
                             vul_ns=bool(vulnerable) and active_side == 0,
                             vul_ew=bool(vulnerable) and active_side == 1)
        return cls(state, active_side)._advance_inactive()

    def _advance_inactive(self) -> "CooperativeAuction":
        state = self.state
        while not state.ended and state.turn % 2 != self.active_side:
            state = state.apply(PASS)
        return CooperativeAuction(state, self.active_side)

    @property
    def ended(self) -> bool:
        return self.state.ended

    @property
    def actor(self) -> int:
        if self.ended:
            raise ValueError("auction is over")
        return self.state.turn

    @property
    def vulnerable(self) -> bool:
        return self.state.vulnerable(self.active_side)

    def legal_mask(self) -> np.ndarray:
        """38-slot mask for the player to act; X/XX always False."""
        mask = self.state.legal_mask()
        mask[DOUBLE] = False
        mask[REDOUBLE] = False
        if not self.ended and self.state.turn % 2 != self.active_side:
            mask[:] = False
            mask[PASS] = True
        return mask

    def apply(self, action: int) -> "CooperativeAuction":
        if action in (DOUBLE, REDOUBLE):
            raise ValueError("Double and Redouble are disabled in the cooperative auction")
        if not 0 <= action < N_ACTIONS or not self.legal_mask()[action]:
            raise ValueError(f"illegal cooperative call {action}")
        return CooperativeAuction(self.state.apply(action), self.active_side)._advance_inactive()

    def endpoint(self, action: int) -> AuctionState:
        """Apply ``action`` then let every remaining player Pass."""
        state = self.apply(action).state
        while not state.ended:
            state = state.apply(PASS)
        return state

    def partnership_score(self, final: AuctionState, tricks: np.ndarray) -> int:
        if final.doubled:
            raise AssertionError("cooperative auction produced a doubled contract")
        ns = terminal_ns_score(final, tricks)
        return ns if self.active_side == 0 else -ns

    def endpoint_score(self, action: int, tricks: np.ndarray) -> int:
        return self.partnership_score(self.endpoint(action), tricks)

    def inactive_calls_are_passes(self) -> bool:
        return all(call == PASS for i, call in enumerate(self.state.calls)
                   if (self.state.dealer + i) % 2 != self.active_side)

    def observation(self, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Actor's own 13 cards and public auction features, nothing else.

        Features: bids by self (35), bids by partner (35), self/partner passed
        before any bid (2), dealer relative to actor (4), own-side vulnerable (1).
        Opponent calls carry no information because they are forced Passes.
        """
        actor = self.actor
        hand = (np.asarray(owners) == actor).astype(np.float32)
        features = np.zeros(AUCTION_FEATURES, dtype=np.float32)
        bid_seen = False
        for i, call in enumerate(self.state.calls):
            seat = (self.state.dealer + i) % 4
            if seat % 2 != self.active_side:
                continue
            mine = seat == actor
            if call < PASS:
                features[(OWN_BIDS if mine else PARTNER_BIDS).start + call] = 1.0
                bid_seen = True
            elif call == PASS and not bid_seen:
                features[SELF_PASSED if mine else PARTNER_PASSED] = 1.0
        features[DEALER_REL.start + (self.state.dealer - actor) % 4] = 1.0
        features[OWN_VUL] = float(self.vulnerable)
        return hand, features
