"""Structured contract finder: categorical DDS tricks plus endpoint Q."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .environment import AUCTION_FEATURES, N_COOP_ACTIONS
from .targets import N_PAIR_ACTIONS

CHECKPOINT_FORMAT = "bridgezero-contract-0.1"
REWARD_UNITS = "(undoubled score - cooperative DD ceiling) / 100"
# Checkpoint label read by the four-seat warm start. The text is kept verbatim so
# checkpoints written by earlier code still load.
STAGE_OBJECTIVES = {
    "D4PG": ("from-scratch endpoint grounding + undiscounted cooperative actor_critic"
             "(terminal_score_minus_dd_ceiling) + trick_ce + chosen_action_q_mse"),
}


class AuctionContractNet(nn.Module):
    """Stage D: own hand + silent-opponent auction features.

    Heads: ``trick_logits (B,2,5,14)`` for actor/partner declaring,
    ``contract_q (B,36)`` endpoint values and ``policy_logits (B,36)`` over
    35 contracts + Pass. There is no Double/Redouble slot.
    """

    kind = "auction"
    inputs = "auction"

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 policy_logit_bound: float | None = None, dropout: float = 0.0):
        super().__init__()
        if policy_logit_bound is not None and policy_logit_bound <= 0:
            raise ValueError("policy_logit_bound must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.policy_logit_bound = policy_logit_bound
        self.config = dict(width=width, suit_width=suit_width, depth=depth)
        if policy_logit_bound is not None:
            self.config["policy_logit_bound"] = policy_logit_bound
        # E52: recorded only when on, so every checkpoint written before it still loads.
        if dropout:
            self.config["dropout"] = dropout

        def drop() -> list[nn.Module]:
            return [nn.Dropout(dropout)] if dropout else []

        self.suit_net = nn.Sequential(
            nn.Linear(13, suit_width), nn.GELU(), *drop(),
            nn.Linear(suit_width, suit_width), nn.GELU(), *drop(),
        )
        self.auction_net = nn.Sequential(
            nn.Linear(AUCTION_FEATURES, width), nn.GELU(), *drop())
        layers: list[nn.Module] = []
        dim = 4 * suit_width + width
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.GELU(), *drop()]
            dim = width
        self.trunk = nn.Sequential(*layers)
        self.trick_head = nn.Linear(width, 2 * 5 * 14)
        self.q_head = nn.Linear(width, N_COOP_ACTIONS)
        self.policy_head = nn.Linear(width, N_COOP_ACTIONS)
        for head in (self.q_head, self.policy_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(hand, auction)
        return {
            "trick_logits": self.trick_head(hidden).view(len(hand), 2, 5, 14),
            "contract_q": self.q_head(hidden),
            "policy_logits": self.policy_logits(hidden),
        }

    def policy_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Policy scores, optionally bounded without changing their ordering."""
        return self.bound_policy_logits(self.policy_head(hidden))

    def bound_policy_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.policy_logit_bound is None:
            return logits
        bound = self.policy_logit_bound
        low = logits.min(-1, keepdim=True).values
        high = logits.max(-1, keepdim=True).values
        center = (low + high) / 2.0
        radius = (high - low) / 2.0
        # Centering is softmax-invariant.  A single positive affine scale keeps
        # every ordering exact while limiting the largest possible logit gap.
        scale = torch.clamp(radius / bound, min=1.0)
        return (logits - center) / scale

    def encode(self, hand: torch.Tensor, auction: torch.Tensor) -> torch.Tensor:
        batch = hand.shape[0]
        encoded = self.suit_net(hand.view(batch, 4, 13)).reshape(batch, -1)
        return self.trunk(torch.cat([encoded, self.auction_net(auction)], dim=-1))

    def shared_parameters(self) -> list[nn.Parameter]:
        return [*self.suit_net.parameters(), *self.auction_net.parameters(),
                *self.trunk.parameters()]


MODEL_KINDS = {"auction": AuctionContractNet}


def save_checkpoint(path: str | Path, net: nn.Module, stage: str, **extra: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": CHECKPOINT_FORMAT, "stage": stage, "objective": STAGE_OBJECTIVES[stage],
                "reward_units": REWARD_UNITS, "model_kind": net.kind,
                "model_config": net.config, "net": net.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if checkpoint.get("reward_units") != REWARD_UNITS:
        raise ValueError("checkpoint reward units differ from this code")
    if checkpoint.get("objective") != STAGE_OBJECTIVES.get(checkpoint.get("stage")):
        raise ValueError(f"{path} has an unknown stage or objective")
    net = MODEL_KINDS[checkpoint["model_kind"]](**checkpoint["model_config"]).to(device)
    net.load_state_dict(checkpoint["net"])
    return net, checkpoint


class CentralCritic(nn.Module):
    """Training-only ``V(pair hands, public auction)`` baseline."""

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3):
        super().__init__()
        self.config = dict(width=width, suit_width=suit_width, depth=depth)
        self.suit_net = nn.Sequential(
            nn.Linear(26, suit_width), nn.GELU(),
            nn.Linear(suit_width, suit_width), nn.GELU())
        self.auction_net = nn.Sequential(nn.Linear(AUCTION_FEATURES, width), nn.GELU())
        layers: list[nn.Module] = []
        dim = 4 * suit_width + width
        for _ in range(depth):
            layers.extend((nn.Linear(dim, width), nn.GELU()))
            dim = width
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(width, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, hands: torch.Tensor, auction: torch.Tensor) -> torch.Tensor:
        batch = len(hands)
        suits = hands.view(batch, 2, 4, 13).transpose(1, 2).reshape(batch, 4, 26)
        encoded = self.suit_net(suits).reshape(batch, -1)
        hidden = self.trunk(torch.cat((encoded, self.auction_net(auction)), dim=-1))
        return self.value_head(hidden).squeeze(-1)
