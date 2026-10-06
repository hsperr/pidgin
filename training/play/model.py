"""The card play net, and the critic that trains it.

Two nets, on purpose:

* ``PlayNet`` sees only what the player on turn may legally see: their own cards, the
  one other hand that is face up, every card already played, the contract and the
  auction. It picks a card and it guesses where the hidden cards are.
* ``Critic`` sees all four hands. That is free in self play and it is never used at
  play time, only to cut the noise out of the policy gradient.

Relative seats are used everywhere: 0 is the player on turn, 1 the next seat round,
2 partner, 3 the seat before. So one net covers all four chairs.
"""

from __future__ import annotations

import torch
from torch import nn

from training.play.data import N_CALLS

N_CARDS = 52
AUCTION_DIM = 96
PLAY_FEATURES = (52 + 52 + 1 + 4 * 52 + 3 * 52 + 5 + 7 + 3 + 3 + 2 + 2 + 1
                 + 4 * N_CALLS + AUCTION_DIM)
CRITIC_FEATURES = 4 * 52 + 3 * 52 + 5 + 7 + 3 + 2 + 2 + 1


def _mlp(inputs: int, width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(inputs, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    return nn.Sequential(*layers)


class AuctionEncoder(nn.Module):
    """Read the auction as a sequence, once per deal, from all four chairs.

    A bag of calls loses the order, and the order is most of the meaning: 1S then 2S
    from partner is a raise, 2S then 1S is impossible. Each call becomes a token that
    carries the call itself and which relative seat made it, and a GRU walks the
    sequence. The auction does not change while the cards are played, so this runs once
    and all 52 decisions reuse it.
    """

    def __init__(self, dim: int = AUCTION_DIM):
        super().__init__()
        self.call_embed = nn.Embedding(N_CALLS + 1, 48, padding_idx=N_CALLS)
        self.seat_embed = nn.Embedding(4, 16)
        self.gru = nn.GRU(48 + 16, dim, batch_first=True)

    def forward(self, calls: torch.Tensor, n_calls: torch.Tensor,
                dealer: torch.Tensor) -> torch.Tensor:
        """``(n, len)`` calls -> ``(n, 4, dim)``, one reading per viewing seat."""
        n, length = calls.shape
        device = calls.device
        caller = (dealer[:, None] + torch.arange(length, device=device)[None, :]) % 4
        tokens = self.call_embed(calls)                                  # (n, len, 48)
        out = []
        for view in range(4):
            seats = _relative(view * torch.ones_like(caller), caller)
            packed = torch.cat([tokens, self.seat_embed(seats)], -1)
            steps, _ = self.gru(packed)
            last = (n_calls - 1).clamp(min=0)
            out.append(steps[torch.arange(n, device=device), last])
        return torch.stack(out, 1)


class PlayNet(nn.Module):
    """Pick a card. Also guess who holds each card you cannot see."""

    def __init__(self, width: int = 512, depth: int = 3):
        super().__init__()
        self.config = dict(width=width, depth=depth)
        self.auction = AuctionEncoder()
        self.trunk = _mlp(PLAY_FEATURES, width, depth)
        self.policy = nn.Linear(width, N_CARDS)
        self.belief = nn.Linear(width, N_CARDS * 4)

    def forward(self, features: torch.Tensor, legal: torch.Tensor) -> dict:
        hidden = self.trunk(features)
        logits = self.policy(hidden)
        logits = logits.masked_fill(~legal, float("-inf"))
        return {
            "log_probs": torch.log_softmax(logits, -1),
            "belief": self.belief(hidden).view(-1, N_CARDS, 4),
        }


class Critic(nn.Module):
    """Tricks the side on turn still takes, judged with every hand face up."""

    def __init__(self, width: int = 384, depth: int = 3):
        super().__init__()
        self.config = dict(width=width, depth=depth)
        self.trunk = _mlp(CRITIC_FEATURES, width, depth)
        self.head = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(features)).squeeze(-1)


def _one_hot(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.zeros(len(index), size, device=index.device).scatter_(1, index[:, None], 1.0)


def _relative(seat: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return (other - seat) % 4


def encode(batch, contracts, dummy_visible: bool, auction: torch.Tensor) -> dict:
    """Features for the seat on turn, one row per deal.

    ``auction`` is ``(n, 4, AUCTION_DIM)`` from :class:`AuctionEncoder`, read once per deal.

    ``dummy_visible`` is False only for the opening lead, when nothing is face up yet.
    The declaring side is treated as one player with two hands: whoever is on turn also
    sees the other of the two, which is exactly what a real declarer sees.
    """
    device = batch.device
    n, turn = batch.n, batch.to_play()
    dummy = (contracts.declarer + 2) % 4

    own = batch.hand(turn).float()
    partner_seat = torch.where(turn == contracts.declarer, dummy,
                               torch.where(turn == dummy, contracts.declarer, dummy))
    face_up = batch.hand(partner_seat).float()
    show = torch.full((n, 1), float(dummy_visible), device=device)
    face_up = face_up * show

    # Every card already played, filed under who played it, in relative seats.
    played_by = torch.zeros(n, 4, N_CARDS, device=device)
    if batch.t > 0:
        cards = batch.history[:, :batch.t]
        rel = _relative(turn[:, None], batch.owner.gather(1, cards))
        rows = torch.arange(n, device=device)[:, None].expand(n, batch.t)
        played_by[rows.reshape(-1), rel.reshape(-1), cards.reshape(-1)] = 1.0

    # The cards already on the table this trick, by how far ahead of me they sat.
    trick = torch.zeros(n, 3, N_CARDS, device=device)
    if batch.pos > 0:
        cards = batch.trick_cards()
        rows = torch.arange(n, device=device)[:, None].expand(n, batch.pos)
        slot = torch.arange(batch.pos, device=device)[None, :].expand(n, batch.pos)
        trick[rows.reshape(-1), (batch.pos - 1 - slot).reshape(-1), cards.reshape(-1)] = 1.0

    my_side = turn % 2
    role = torch.stack([turn == contracts.declarer, turn == dummy,
                        (turn % 2) != (contracts.declarer % 2)], 1).float()
    vul = torch.stack([torch.where(my_side == 0, contracts.vul_ns, contracts.vul_ew),
                       torch.where(my_side == 0, contracts.vul_ew, contracts.vul_ns)], 1).float()
    won = batch.tricks_won.float() / 13.0
    mine = torch.stack([won[torch.arange(n, device=device), my_side],
                        won[torch.arange(n, device=device), 1 - my_side]], 1)
    # Both readings of the auction, rotated so slot 0 is always the player on turn:
    # the bag (which calls each relative seat made, the bidding net's own input) and the
    # sequence reading, which is the only one that knows what order they came in.
    seats_round = (turn[:, None] + torch.arange(4, device=device)[None, :]) % 4
    bag = contracts.call_bag.gather(1, seats_round[:, :, None].expand(n, 4, N_CALLS))
    heard = auction[torch.arange(n, device=device), turn]

    features = torch.cat([
        own, face_up, show,
        played_by.reshape(n, -1), trick.reshape(n, -1),
        _one_hot(contracts.trump, 5), _one_hot(contracts.level - 1, 7),
        _one_hot(contracts.doubled, 3), role, vul, mine,
        torch.full((n, 1), batch.trick_no / 13.0, device=device),
        bag.reshape(n, -1), heard,
    ], 1)

    all_hands = torch.stack([batch.hand((turn + k) % 4).float() for k in range(4)], 1)
    critic = torch.cat([
        all_hands.reshape(n, -1), trick.reshape(n, -1),
        _one_hot(contracts.trump, 5), _one_hot(contracts.level - 1, 7),
        _one_hot(contracts.doubled, 3), vul, mine,
        torch.full((n, 1), batch.trick_no / 13.0, device=device),
    ], 1)

    # Belief target: for every card I cannot see, which relative seat holds it.
    seen = own.bool() | face_up.bool() | ~batch.unplayed
    target = _relative(turn[:, None], batch.owner)
    return {"features": features, "critic": critic, "turn": turn,
            "belief_target": target, "belief_mask": ~seen}
