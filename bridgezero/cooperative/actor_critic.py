"""On-policy trajectories and losses for silent-opponent cooperative auctions.

The deployable actor sees only its hand and the public auction.  The critic is a
centralized-training-only baseline that sees both partnership hands.  Opponents,
legality, declarer semantics, and terminal scoring all reuse ``contract`` code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..contract.data import TorchDeals
from ..contract.environment import AUCTION_FEATURES, N_COOP_ACTIONS
from ..contract.prefixes import CoopBatch, final_scores, net_outputs
from ..contract.targets import TARGET_SCALE, TorchScorer


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


@dataclass
class Trajectories:
    """A batch of complete episodes and their pre-action information states."""

    states: CoopBatch
    actions: torch.Tensor
    episode: torch.Tensor
    terminal: CoopBatch
    score: torch.Tensor
    ceiling: torch.Tensor

    @property
    def returns(self) -> torch.Tensor:
        return (self.score - self.ceiling) / TARGET_SCALE

    def __len__(self) -> int:
        return len(self.actions)


def mix_legal_uniform(probs: torch.Tensor, legal: torch.Tensor,
                      uniform_mix: float) -> torch.Tensor:
    """Add explicit support to every legal call without changing legality."""
    if not 0.0 <= uniform_mix < 1.0:
        raise ValueError("uniform_mix must be in [0, 1)")
    if probs.shape != legal.shape:
        raise ValueError("policy and legal mask shapes differ")
    uniform = legal.float() / legal.sum(-1, keepdim=True).clamp(min=1)
    return (1.0 - uniform_mix) * probs + uniform_mix * uniform


@torch.no_grad()
def collect_trajectories(actor, deals: TorchDeals, episodes: int,
                         generator: torch.Generator, scorer: TorchScorer,
                         temperature: float = 1.0, device: torch.device | str = "cpu",
                         max_decisions: int = 40,
                         uniform_mix: float = 0.0) -> Trajectories:
    """Sample complete auctions from the current policy.

    Sampling occurs on CPU with ``generator`` so identical seeds produce the same
    actions on CPU and MPS.  Only the existing 36-slot legal mask is sampled;
    Double and Redouble are structurally absent.
    """
    if episodes < 1 or temperature <= 0:
        raise ValueError("episodes and temperature must be positive")
    if not 0.0 <= uniform_mix < 1.0:
        raise ValueError("uniform_mix must be in [0, 1)")
    device = torch.device(device)

    def rand(high: int) -> torch.Tensor:
        return torch.randint(high, (episodes,), generator=generator).to(device)

    terminal = CoopBatch.start(rand(deals.n), rand(2), rand(4), rand(2))
    states: list[CoopBatch] = []
    actions: list[torch.Tensor] = []
    episode_rows: list[torch.Tensor] = []
    actor.eval()
    for _ in range(max_decisions):
        alive = ~terminal.ended
        if not bool(alive.any()):
            break
        ids = alive.nonzero().squeeze(1)
        state = terminal.subset(alive)
        legal = state.legal()
        logits = net_outputs(actor, deals, state)["policy_logits"] / temperature
        probs = torch.softmax(logits.masked_fill(~legal, -1e9), -1)
        probs = mix_legal_uniform(probs, legal, uniform_mix)
        chosen = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).to(device)
        if not bool(legal[torch.arange(len(state), device=device), chosen].all()):
            raise AssertionError("on-policy sampler produced an illegal call")
        states.append(state)
        actions.append(chosen)
        episode_rows.append(ids)
        full_action = torch.zeros(episodes, dtype=torch.long, device=device)
        full_action[alive] = chosen
        terminal.apply(full_action, alive)
    if not bool(terminal.ended.all()):
        raise RuntimeError(f"cooperative auction exceeded {max_decisions} active decisions")
    score, ceiling, _ = final_scores(terminal, deals, scorer)
    return Trajectories(CoopBatch.cat(states), torch.cat(actions), torch.cat(episode_rows),
                        terminal, score, ceiling)


def trajectory_losses(actor, critic: CentralCritic, deals: TorchDeals,
                      trajectories: Trajectories, entropy_weight: float = 0.01,
                      policy_temperature: float = 1.0) -> dict:
    """Undiscounted team-return actor/critic and auxiliary outcome losses.

    Policy and entropy terms are summed within each episode and then averaged over
    episodes.  This is the episodic ``gamma=1`` objective rather than weighting
    long auctions merely because they contain more stored decisions.
    """
    if policy_temperature <= 0:
        raise ValueError("policy temperature must be positive")
    states = trajectories.states
    out = net_outputs(actor, deals, states)
    legal = states.legal()
    logits = out["policy_logits"] / policy_temperature
    log_policy = F.log_softmax(logits.masked_fill(~legal, -1e9), -1)
    policy = log_policy.exp()
    row = torch.arange(len(states), device=states.deal.device)
    chosen_logp = log_policy[row, trajectories.actions]
    entropy = -(policy * log_policy.masked_fill(~legal, 0.0)).sum(-1)

    actor_seat = states.actor_seat
    pair = torch.stack((deals.hands[states.deal, actor_seat],
                        deals.hands[states.deal, (actor_seat + 2) % 4]), dim=1)
    value = critic(pair, states.features())
    returns = trajectories.returns[trajectories.episode]
    advantage = returns - value.detach()
    episodes = len(trajectories.score)
    policy_by_episode = torch.zeros(episodes, device=returns.device).scatter_add_(
        0, trajectories.episode, chosen_logp * advantage)
    entropy_by_episode = torch.zeros(episodes, device=returns.device).scatter_add_(
        0, trajectories.episode, entropy)

    rel = deals.rel_tricks(states.deal, actor_seat)
    return {
        "policy_loss": -policy_by_episode.mean(),
        "entropy": entropy_by_episode.mean(),
        "policy_objective": -policy_by_episode.mean() - entropy_weight * entropy_by_episode.mean(),
        "critic_loss": F.mse_loss(value, returns),
        "q_loss": F.mse_loss(out["contract_q"][row, trajectories.actions], returns),
        "trick_nll": F.cross_entropy(out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
        "return_mean": trajectories.returns.mean(),
        "advantage_mean": advantage.mean(),
        "advantage_std": advantage.std(),
    }
