"""Structured contract finder: categorical DDS tricks plus endpoint Q."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import INPUT_HANDS
from .environment import AUCTION_FEATURES, N_COOP_ACTIONS
from .prefixes import OBSERVATIONS
from .targets import N_PAIR_ACTIONS

CHECKPOINT_FORMAT = "bridgezero-contract-0.1"
REWARD_UNITS = "(undoubled score - cooperative DD ceiling) / 100"
# Stage D policy distillation targets. The first source of each stage is its default;
# checkpoints written before the source was recorded (E4, E5 D2 runs) used it.
POLICY_TARGETS = {"expected": "target_net expected_endpoint", "q": "target_net contract_q",
                  "rollout": "exact frozen-policy continuation values",
                  "exact_pg": "exact frozen-policy continuation values"}
STAGE_POLICY_SOURCES = {"D1": ("expected",), "D2": ("q", "expected"),
                        "D4": ("rollout", "q", "exact_pg")}
_POLICY_CE = " + policy_ce(softmax({} / policy_temperature))"
# exact_pg maximizes the expected value of the policy's own call distribution over
# every legal call. Averaged over hidden deals this is E[value | observation], so its
# optimum is softmax(conditional value / temperature), not a per-deal best-call label.
_POLICY_PG = (" + policy_value(-sum_a pi(a) {} - policy_temperature * entropy(pi)"
              " + signal_weight * sum_a pi(a) listener_partner_nats(a))")
Q_LOSSES = ("huber", "mse")
STAGE_OBJECTIVES = {
    "B": "trick_ce + huber(pair_endpoint_q_71)",
    "C": "trick_ce + huber(pair_endpoint_q_71)",
    "D1": "trick_ce + huber(legal_endpoint_q_36)" + _POLICY_CE,
    "D2": ("trick_ce + huber(legal_q_36: endpoint rows, or continuation rows finished by the "
           "greedy target-net policy)" + _POLICY_CE),
    "D4": ("trick_ce + partner_card_bce/count + huber(legal_q_36: all roots finished by "
           "a frozen greedy policy)" + _POLICY_CE),
    "D4PG": ("from-scratch endpoint grounding + undiscounted cooperative actor_critic"
             "(terminal_score_minus_dd_ceiling) + trick_ce + chosen_action_q_mse"),
    "D4JPS": ("from-scratch endpoint grounding + exact two-ply joint-policy expectation "
              "+ frozen-blueprint full-auction continuation"),
    "D4CF": ("from-scratch endpoint grounding + all-legal-call frozen-policy continuation; "
             "Q_continuation = expected_endpoint_from_tricks + learned_zero_init_residual; "
             "policy_ce from conditional target-net Q with legal-uniform support floor"),
}
# Every checkpoint written before per-stage objectives (E1-E4, including the
# frozen D1 run) carries this string, which is wrong for D1. It is corrected in
# memory at load time; the files themselves are never rewritten.
LEGACY_OBJECTIVE = "trick_ce + huber(pair_endpoint_q)"


def default_policy_source(stage: str) -> str | None:
    return STAGE_POLICY_SOURCES[stage][0] if stage in STAGE_POLICY_SOURCES else None


def objective_for(stage: str, policy_source: str | None = None, q_loss: str = "huber") -> str:
    """Objective text; Stage D names its policy target (``None``: the stage default)."""
    if stage not in STAGE_OBJECTIVES:
        raise ValueError(f"no objective is defined for stage {stage!r}")
    if q_loss not in Q_LOSSES:
        raise ValueError(f"unknown Q loss {q_loss!r}")
    text = STAGE_OBJECTIVES[stage]
    if stage not in STAGE_POLICY_SOURCES:
        if policy_source is not None:
            raise ValueError(f"stage {stage} has no policy target")
    else:
        source = policy_source or default_policy_source(stage)
        if source not in STAGE_POLICY_SOURCES[stage]:
            raise ValueError(f"policy source {source!r} is not allowed in stage {stage}")
        if source == "exact_pg":
            text = text.replace(_POLICY_CE, _POLICY_PG)
        text = text.format(POLICY_TARGETS[source])
    return text.replace("huber(", f"{q_loss}(")


class ContractNet(nn.Module):
    """Hands -> ``trick_logits (B,2,5,14)`` and ``contract_q (B,71)``.

    Relative declarers are 0=actor, 1=partner; strains are table order S,H,D,C,NT.
    ``contract_q`` is indexed ``declarer * 35 + contract`` with Pass at 70.
    The same architecture serves Stage B (both hands) and Stage C (own hand).
    """

    kind = "hand"

    def __init__(self, inputs: str = "single", width: int = 256,
                 suit_width: int = 64, depth: int = 3):
        super().__init__()
        n_hands = INPUT_HANDS[inputs]
        self.inputs = inputs
        self.config = dict(inputs=inputs, width=width, suit_width=suit_width, depth=depth)
        # One shared per-suit encoder; suit position is kept when concatenated.
        self.suit_net = nn.Sequential(
            nn.Linear(13 * n_hands, suit_width), nn.GELU(),
            nn.Linear(suit_width, suit_width), nn.GELU(),
        )
        layers: list[nn.Module] = []
        dim = 4 * suit_width + 1
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.GELU()]
            dim = width
        self.trunk = nn.Sequential(*layers)
        self.trick_head = nn.Linear(width, 2 * 5 * 14)
        self.q_head = nn.Linear(width, N_PAIR_ACTIONS)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, hands: torch.Tensor, vulnerable: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, n_hands, _ = hands.shape
        suits = hands.view(batch, n_hands, 4, 13).transpose(1, 2).reshape(batch, 4, -1)
        encoded = self.suit_net(suits).reshape(batch, -1)
        hidden = self.trunk(torch.cat([encoded, vulnerable.float().unsqueeze(-1)], dim=-1))
        return {
            "trick_logits": self.trick_head(hidden).view(batch, 2, 5, 14),
            "contract_q": self.q_head(hidden),
        }

    def shared_parameters(self) -> list[nn.Parameter]:
        return [*self.suit_net.parameters(), *self.trunk.parameters()]


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


class ContinuationResidualAuctionNet(AuctionContractNet):
    """Auction bidder with a zero-initialized continuation residual for every call.

    ``continuation_residual`` is trained to predict the additional value obtained
    after a call when the partnership keeps bidding, relative to ending at that
    call.  Combining it with the structured trick-distribution endpoint value
    keeps Pass and contract values on one calibrated scale.
    """

    kind = "continuation_residual_auction"

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 policy_logit_bound: float | None = None):
        super().__init__(width, suit_width, depth, policy_logit_bound)
        self.continuation_head = nn.Linear(width, N_COOP_ACTIONS)
        nn.init.zeros_(self.continuation_head.weight)
        nn.init.zeros_(self.continuation_head.bias)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(hand, auction)
        return {
            "trick_logits": self.trick_head(hidden).view(len(hand), 2, 5, 14),
            "contract_q": self.q_head(hidden),
            "continuation_residual": self.continuation_head(hidden),
            "policy_logits": self.policy_logits(hidden),
        }


class BeliefAuctionNet(nn.Module):
    """Full-auction model whose value heads consume an explicit partner belief.

    The belief is inferred only from the actor's hand and public auction. Known
    own cards are zeroed before the soft partner hand is encoded. Trick, Q, and
    policy losses therefore also train the belief pathway, while the supervised
    belief head supplies dense communication credit at every auction prefix.
    """

    kind = "belief_auction"
    inputs = "auction"

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 belief_width: int = 192):
        super().__init__()
        self.config = dict(width=width, suit_width=suit_width, depth=depth,
                           belief_width=belief_width)
        self.own_suit_net = nn.Sequential(
            nn.Linear(13, suit_width), nn.GELU(),
            nn.Linear(suit_width, suit_width), nn.GELU())
        self.auction_net = nn.Sequential(nn.Linear(AUCTION_FEATURES, width), nn.GELU())
        self.belief_body = nn.Sequential(
            nn.Linear(4 * suit_width + width, belief_width), nn.GELU(),
            nn.Linear(belief_width, belief_width), nn.GELU())
        self.belief_head = nn.Linear(belief_width, 52)
        # Same per-suit structure as Stage B, except the second hand is a
        # differentiable probability vector rather than a revealed hand.
        self.pair_suit_net = nn.Sequential(
            nn.Linear(26, suit_width), nn.GELU(),
            nn.Linear(suit_width, suit_width), nn.GELU())
        layers: list[nn.Module] = []
        dim = 4 * suit_width + width + belief_width
        for _ in range(depth):
            layers += [nn.Linear(dim, width), nn.GELU()]
            dim = width
        self.trunk = nn.Sequential(*layers)
        self.trick_head = nn.Linear(width, 2 * 5 * 14)
        self.q_head = nn.Linear(width, N_COOP_ACTIONS)
        self.policy_head = nn.Linear(width, N_COOP_ACTIONS)
        for head in (self.q_head, self.policy_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = len(hand)
        own = self.own_suit_net(hand.view(batch, 4, 13)).reshape(batch, -1)
        public = self.auction_net(auction)
        belief_hidden = self.belief_body(torch.cat((own, public), dim=-1))
        belief_logits = self.belief_head(belief_hidden)
        partner_probability = torch.sigmoid(belief_logits) * (1.0 - hand)
        pair = torch.stack((hand, partner_probability), dim=1)
        pair_suits = pair.view(batch, 2, 4, 13).transpose(1, 2).reshape(batch, 4, 26)
        pair_encoded = self.pair_suit_net(pair_suits).reshape(batch, -1)
        hidden = self.trunk(torch.cat((pair_encoded, public, belief_hidden), dim=-1))
        return {
            "partner_belief_logits": belief_logits,
            "partner_probability": partner_probability,
            "trick_logits": self.trick_head(hidden).view(batch, 2, 5, 14),
            "contract_q": self.q_head(hidden),
            "policy_logits": self.policy_head(hidden),
        }

    def shared_parameters(self) -> list[nn.Parameter]:
        return [*self.own_suit_net.parameters(), *self.auction_net.parameters(),
                *self.belief_body.parameters(), *self.belief_head.parameters(),
                *self.pair_suit_net.parameters(), *self.trunk.parameters()]


class ResidualBeliefAuctionNet(AuctionContractNet):
    """Belief adapter that is exactly a pretrained auction net at initialization."""

    kind = "residual_belief_auction"

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 belief_width: int = 192):
        super().__init__(width, suit_width, depth)
        self.config["belief_width"] = belief_width
        self.belief_body = nn.Sequential(
            nn.Linear(width, belief_width), nn.GELU(),
            nn.Linear(belief_width, belief_width), nn.GELU())
        self.belief_head = nn.Linear(belief_width, 52)
        self.partner_suit_net = nn.Sequential(
            nn.Linear(13, suit_width), nn.GELU(),
            nn.Linear(suit_width, suit_width), nn.GELU())
        self.belief_adapter = nn.Sequential(
            nn.Linear(width + belief_width + 4 * suit_width, width), nn.GELU(),
            nn.Linear(width, width))
        nn.init.zeros_(self.belief_adapter[-1].weight)
        nn.init.zeros_(self.belief_adapter[-1].bias)

    @torch.no_grad()
    def load_base(self, base: AuctionContractNet) -> None:
        expected = {k: self.config[k] for k in ("width", "suit_width", "depth")}
        if base.config != expected:
            raise ValueError(f"base architecture {base.config} != {expected}")
        own = self.state_dict()
        for name, value in base.state_dict().items():
            own[name].copy_(value)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = len(hand)
        base_hidden = self.encode(hand, auction)
        belief_hidden = self.belief_body(base_hidden)
        belief_logits = self.belief_head(belief_hidden)
        partner_probability = torch.sigmoid(belief_logits) * (1.0 - hand)
        partner_encoded = self.partner_suit_net(
            partner_probability.view(batch, 4, 13)).reshape(batch, -1)
        residual = self.belief_adapter(torch.cat(
            (base_hidden, belief_hidden, partner_encoded), dim=-1))
        hidden = base_hidden + residual
        return {
            "partner_belief_logits": belief_logits,
            "partner_probability": partner_probability,
            "trick_logits": self.trick_head(hidden).view(batch, 2, 5, 14),
            "contract_q": self.q_head(hidden),
            "policy_logits": self.policy_head(hidden),
        }

    def freeze_base(self) -> None:
        """Freeze the exact pretrained policy; leave only belief/residual modules trainable."""
        for module in (self.suit_net, self.auction_net, self.trunk,
                       self.trick_head, self.q_head, self.policy_head):
            module.requires_grad_(False)

    def shared_parameters(self) -> list[nn.Parameter]:
        return [*super().shared_parameters(), *self.belief_body.parameters(),
                *self.belief_head.parameters(), *self.partner_suit_net.parameters(),
                *self.belief_adapter.parameters()]


MODEL_KINDS = {"hand": ContractNet, "auction": AuctionContractNet,
               "continuation_residual_auction": ContinuationResidualAuctionNet,
               "belief_auction": BeliefAuctionNet,
               "residual_belief_auction": ResidualBeliefAuctionNet}


def save_checkpoint(path: str | Path, net: nn.Module, stage: str,
                    policy_source: str | None = None, q_loss: str = "huber",
                    **extra: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = policy_source or default_policy_source(stage)
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "objective": objective_for(stage, source, q_loss),
        **({"policy_source": source} if source else {}),
        # Checkpoints without the field were trained with Huber Q regression.
        **({"q_loss": q_loss} if q_loss != "huber" else {}),
        "reward_units": REWARD_UNITS,
        "model_kind": net.kind,
        "model_config": net.config,
        "net": net.state_dict(),
        **extra,
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu",
                    stage: str | None = None) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if checkpoint.get("reward_units") != REWARD_UNITS:
        raise ValueError("checkpoint reward units differ from this code")
    if stage is not None and checkpoint.get("stage") != stage:
        raise ValueError(f"checkpoint stage {checkpoint.get('stage')} != {stage}")
    # Stage D checkpoints written before the policy source was recorded used the stage default.
    source = default_policy_source(checkpoint.get("stage"))
    if source:
        source = checkpoint.setdefault("policy_source", source)
    expected = objective_for(checkpoint.get("stage"), source,
                             checkpoint.setdefault("q_loss", "huber"))
    if checkpoint.get("objective") == LEGACY_OBJECTIVE:
        checkpoint["stored_objective"] = LEGACY_OBJECTIVE
        checkpoint["objective"] = expected
    elif checkpoint.get("objective") != expected:
        raise ValueError(f"checkpoint objective {checkpoint.get('objective')!r} "
                         f"is not the stage {checkpoint.get('stage')} objective")
    # Checkpoints written before observation regimes (E1-E4) saw intact features.
    checkpoint.setdefault("observation", "intact")
    if checkpoint["observation"] not in OBSERVATIONS:
        raise ValueError(f"unknown observation regime {checkpoint['observation']!r}")
    net = MODEL_KINDS[checkpoint.get("model_kind", "hand")](**checkpoint["model_config"]).to(device)
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
