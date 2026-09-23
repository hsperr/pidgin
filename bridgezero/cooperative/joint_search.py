"""Auditable two-ply joint policy improvement for cooperative bidding.

For every legal call by the current player, this module enumerates every legal
reply by partner.  A frozen blueprint finishes each branch.  The learner then
maximizes the exact expectation over both decentralized policies at once.  DDS
scores construct detached training targets; neither actor invocation sees DDS or
the partner's hand.

This is the first, deliberately bounded JPS-style operator.  It changes two
linked decisions together and leaves deeper continuation to a short-lived
blueprint.  It is kept separate from ordinary trajectory policy gradient so its
cost and lift can be measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..contract.data import TorchDeals
from ..contract.evaluate_auction import continue_auctions
from ..contract.prefixes import CoopBatch, decision_values, final_scores, net_outputs
from ..contract.targets import TARGET_SCALE, TorchScorer


@dataclass
class JointBranches:
    """Detached rollout table for a batch of joint-decision roots."""

    roots: CoopBatch
    root_row: torch.Tensor
    root_action: torch.Tensor
    root_edge_return: torch.Tensor
    listeners: CoopBatch
    listener_root_edge: torch.Tensor
    reply_listener_row: torch.Tensor
    reply_action: torch.Tensor
    reply_return: torch.Tensor
    root_ceiling: torch.Tensor

    @property
    def branch_count(self) -> int:
        return len(self.reply_action) + int(torch.isfinite(self.root_edge_return).sum())


def sample_opening_roots(deals: TorchDeals, n: int, generator: torch.Generator,
                         device: torch.device | str = "cpu") -> CoopBatch:
    """Randomized opening states; the actor still has all 36 legal calls."""
    if n < 1:
        raise ValueError("joint-search roots must be positive")
    device = torch.device(device)

    def rand(high: int) -> torch.Tensor:
        return torch.randint(high, (n,), generator=generator).to(device)

    return CoopBatch.start(rand(deals.n), rand(2), rand(4), rand(2))


@torch.no_grad()
def run_n_ply(actor, deals: TorchDeals, roots: CoopBatch, scorer: TorchScorer,
              decisions: int, rule: str = "policy") -> CoopBatch:
    """Play ``decisions`` active calls greedily, then force partnership Passes."""
    if decisions < 0:
        raise ValueError("decisions must be non-negative")
    batch = roots
    for _ in range(decisions):
        alive = ~batch.ended
        if not bool(alive.any()):
            break
        sub = batch.subset(alive)
        values = decision_values(actor, deals, sub, scorer, rule)
        chosen = values.masked_fill(~sub.legal(), -torch.inf).argmax(-1)
        action = torch.zeros(len(batch), dtype=torch.long, device=batch.deal.device)
        action[alive] = chosen
        batch.apply(action, alive)
    while not bool(batch.ended.all()):
        alive = ~batch.ended
        forced_pass = torch.full(
            (len(batch),), 35, dtype=torch.long, device=batch.deal.device)
        batch.apply(forced_pass, alive)
    return batch


def run_two_ply(actor, deals: TorchDeals, roots: CoopBatch, scorer: TorchScorer,
                rule: str = "policy") -> CoopBatch:
    return run_n_ply(actor, deals, roots, scorer, 2, rule)


@torch.no_grad()
def build_joint_branches(blueprint, deals: TorchDeals, roots: CoopBatch,
                         scorer: TorchScorer, continuation_rule: str = "policy") -> JointBranches:
    """Enumerate current call x partner reply, then finish or stop each branch.

    ``continuation_rule='stop'`` forces the partnership to Pass after the two
    searched decisions and therefore needs no blueprint.  Other rules use the
    frozen blueprint for the remaining full auction.
    """
    if continuation_rule != "stop" and blueprint is None:
        raise ValueError("a blueprint is required for joint-search continuation")
    if bool(roots.ended.any()):
        raise ValueError("joint-search roots must be live")
    legal = roots.legal()
    root_row, root_action = legal.nonzero(as_tuple=True)
    after_root = roots.subset(root_row)
    after_root.apply(root_action, torch.ones_like(root_row, dtype=torch.bool))

    root_edge_return = torch.full(
        (len(root_row),), torch.nan, dtype=torch.float32, device=root_row.device)
    terminal = after_root.ended
    if bool(terminal.any()):
        score, ceiling, _ = final_scores(after_root.subset(terminal), deals, scorer)
        root_edge_return[terminal] = (score - ceiling) / TARGET_SCALE

    listener_root_edge = (~terminal).nonzero().squeeze(1)
    listeners = after_root.subset(~terminal)
    reply_listener_row, reply_action = listeners.legal().nonzero(as_tuple=True)
    replies = listeners.subset(reply_listener_row)
    replies.apply(reply_action, torch.ones_like(reply_listener_row, dtype=torch.bool))
    if continuation_rule == "stop":
        alive = ~replies.ended
        forced_pass = torch.full(
            (len(replies),), 35, dtype=torch.long, device=reply_listener_row.device)
        replies.apply(forced_pass, alive)
        if not bool(replies.ended.all()):
            raise AssertionError("two-ply forced-stop branch did not terminate")
    else:
        continue_auctions(blueprint, deals, replies, scorer, continuation_rule)
    reply_score, reply_ceiling, _ = final_scores(replies, deals, scorer)
    reply_return = (reply_score - reply_ceiling) / TARGET_SCALE

    # Ceiling is action-independent and is logged only to translate the joint
    # regret objective back into ordinary duplicate-score units.
    # Compute one ceiling per root without manufacturing a terminal auction.
    rel = deals.rel_tricks(roots.deal, roots.a0)
    root_ceiling = scorer.ceiling(scorer.exact(rel, roots.vul))
    return JointBranches(roots, root_row, root_action, root_edge_return, listeners,
                         listener_root_edge, reply_listener_row, reply_action,
                         reply_return, root_ceiling)


def joint_objective_from_logits(root_logits: torch.Tensor, listener_logits: torch.Tensor,
                                branches: JointBranches, entropy_weight: float = 0.01,
                                temperature: float = 1.0) -> dict[str, torch.Tensor]:
    """Exact two-policy expectation for a fixed detached branch table."""
    if temperature <= 0:
        raise ValueError("joint-search temperature must be positive")
    root_legal = branches.roots.legal()
    root_logp = F.log_softmax(
        (root_logits / temperature).masked_fill(~root_legal, -1e9), -1)
    root_policy = root_logp.exp()
    root_entropy = -(root_policy * root_logp.masked_fill(~root_legal, 0.0)).sum(-1)
    root_edge_probability = root_policy[branches.root_row, branches.root_action]
    edge_value = branches.root_edge_return.clone()
    edge_listener_entropy = torch.zeros_like(edge_value)

    if len(branches.listeners):
        listener_legal = branches.listeners.legal()
        listener_logp = F.log_softmax(
            (listener_logits / temperature).masked_fill(~listener_legal, -1e9), -1)
        listener_policy = listener_logp.exp()
        listener_entropy = -(
            listener_policy * listener_logp.masked_fill(~listener_legal, 0.0)).sum(-1)
        reply_probability = listener_policy[
            branches.reply_listener_row, branches.reply_action]
        listener_value = torch.zeros(
            len(branches.listeners), device=root_logits.device).scatter_add_(
                0, branches.reply_listener_row,
                reply_probability * branches.reply_return)
        edge_value[branches.listener_root_edge] = listener_value
        edge_listener_entropy[branches.listener_root_edge] = listener_entropy

    if not bool(torch.isfinite(edge_value).all()):
        raise AssertionError("a joint-search root edge has no terminal value")
    root_value = torch.zeros(len(branches.roots), device=root_logits.device).scatter_add_(
        0, branches.root_row, root_edge_probability * edge_value)
    expected_listener_entropy = torch.zeros_like(root_value).scatter_add_(
        0, branches.root_row, root_edge_probability * edge_listener_entropy)
    joint_entropy = root_entropy + expected_listener_entropy
    objective = root_value.mean() + entropy_weight * joint_entropy.mean()
    mean_score = root_value.mean() * TARGET_SCALE + branches.root_ceiling.mean()
    return {
        "loss": -objective,
        "return": root_value.mean(),
        "score": mean_score,
        "root_entropy": root_entropy.mean(),
        "listener_entropy": expected_listener_entropy.mean(),
    }


def joint_policy_loss(actor, deals: TorchDeals, branches: JointBranches,
                      entropy_weight: float = 0.01,
                      temperature: float = 1.0) -> dict[str, torch.Tensor]:
    """Evaluate both actor roles using only their legal decentralized inputs."""
    root_logits = net_outputs(actor, deals, branches.roots)["policy_logits"]
    listener_logits = (net_outputs(actor, deals, branches.listeners)["policy_logits"]
                       if len(branches.listeners)
                       else root_logits.new_empty((0, root_logits.shape[1])))
    return joint_objective_from_logits(
        root_logits, listener_logits, branches, entropy_weight, temperature)
