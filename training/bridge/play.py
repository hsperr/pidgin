"""Batched card play: thirteen tricks for many deals at once.

Card index is ``suit * 13 + rank`` with suits S,H,D,C and rank 0 = ace, 12 = two,
the same convention as :mod:`training.bridge.deals`. So the *lowest* rank index
wins a trick. Strain 0..3 is a trump suit, 4 is notrump, matching the double dummy
table's S,H,D,C,NT column order.

Every deal plays exactly 52 cards, so a batch moves in lockstep: step ``t`` is
trick ``t // 4``, position ``t % 4``. No padding and no per-row branching.
"""

from __future__ import annotations

import torch

N_CARDS, N_SEATS = 52, 4
SUIT_OF = torch.arange(N_CARDS) // 13
RANK_OF = torch.arange(N_CARDS) % 13


class PlayBatch:
    """``n`` deals being played out, all at the same card number."""

    def __init__(self, owner: torch.Tensor, trump: torch.Tensor, declarer: torch.Tensor):
        """``owner[i, card]`` is the seat holding it; ``trump`` 0..4; ``declarer`` 0..3."""
        device = owner.device
        self.n = len(owner)
        self.device = device
        self.owner = owner.to(torch.long)
        self.unplayed = torch.ones(self.n, N_CARDS, dtype=torch.bool, device=device)
        self.trump = trump.to(torch.long)
        self.declarer = declarer.to(torch.long)
        self.leader = (self.declarer + 1) % N_SEATS       # opening lead sits left of declarer
        self.t = 0
        self.tricks_won = torch.zeros(self.n, 2, dtype=torch.long, device=device)
        self.history = torch.full((self.n, N_CARDS), -1, dtype=torch.long, device=device)
        self.trick_winner = torch.full((self.n, 13), -1, dtype=torch.long, device=device)
        self.suit_of = SUIT_OF.to(device)
        self.rank_of = RANK_OF.to(device)
        self.rows = torch.arange(self.n, device=device)

    # -- where we are ---------------------------------------------------

    @property
    def pos(self) -> int:
        """Seats already played in the current trick, 0..3."""
        return self.t % N_SEATS

    @property
    def trick_no(self) -> int:
        return self.t // N_SEATS

    @property
    def done(self) -> bool:
        return self.t == N_CARDS

    def to_play(self) -> torch.Tensor:
        return (self.leader + self.pos) % N_SEATS

    def trick_cards(self) -> torch.Tensor:
        """``(n, pos)`` cards played so far in the current trick, in play order."""
        start = self.trick_no * N_SEATS
        return self.history[:, start:start + self.pos]

    def led_suit(self) -> torch.Tensor:
        """``(n,)`` suit led this trick, or -1 when nobody has led yet."""
        if self.pos == 0:
            return torch.full((self.n,), -1, dtype=torch.long, device=self.device)
        return self.suit_of[self.history[:, self.trick_no * N_SEATS]]

    # -- moves ----------------------------------------------------------

    def hand(self, seat: torch.Tensor) -> torch.Tensor:
        """``(n, 52)`` bool: cards ``seat[i]`` still holds."""
        return (self.owner == seat[:, None]) & self.unplayed

    def legal(self) -> torch.Tensor:
        """``(n, 52)`` bool: cards the seat on turn may play. You must follow suit."""
        hand = self.hand(self.to_play())
        if self.pos == 0:
            return hand
        follow = hand & (self.suit_of[None, :] == self.led_suit()[:, None])
        return torch.where(follow.any(1, keepdim=True), follow, hand)

    def play(self, card: torch.Tensor) -> None:
        """Play one card per deal. Resolves the trick when the fourth card lands."""
        if self.done:
            raise RuntimeError("the deal is over")
        self.unplayed[self.rows, card] = False
        self.history[:, self.t] = card
        self.t += 1
        if self.pos == 0:
            self._finish_trick()

    def _finish_trick(self) -> None:
        start = (self.trick_no - 1) * N_SEATS
        trick = self.history[:, start:start + N_SEATS]              # (n, 4) in play order
        suits, ranks = self.suit_of[trick], self.rank_of[trick]
        led = suits[:, :1]
        # No-trump needs no special case: strain 4 never equals a suit index 0..3, so
        # `is_trump` is all false and the led suit decides. Trumps beat everything else.
        is_trump = suits == self.trump[:, None]
        contends = torch.where(is_trump.any(1, keepdim=True), is_trump, suits == led)
        best = torch.where(contends, ranks, torch.full_like(ranks, 99)).argmin(1)
        winner = (self.leader + best) % N_SEATS
        self.trick_winner[:, self.trick_no - 1] = winner
        self.tricks_won[self.rows, winner % 2] += 1
        self.leader = winner

    # -- results --------------------------------------------------------

    def declarer_tricks(self) -> torch.Tensor:
        return self.tricks_won[self.rows, self.declarer % 2]


def play_random(batch: PlayBatch, generator: torch.Generator | None = None) -> PlayBatch:
    """Play every remaining card at random. The floor every net must beat."""
    while not batch.done:
        legal = batch.legal().float()
        batch.play(torch.multinomial(legal, 1, generator=generator).squeeze(1))
    return batch
