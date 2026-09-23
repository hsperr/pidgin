"""Losses for the four-seat own-bid game with Double (stage D5OWN4X).

Reward design (fixed by the user):

- every side's calls get its undoubled own-bid return; being doubled never
  charges the bidding side;
- the Double that stands at auction end on the opponents' final contract gets an
  extra ``delta = (defenders' doubled - undoubled result) / 100`` on that action's
  advantage only (and on its Q target), not on the side's other calls;
- counterfactual: at decisions where the actor may double an undoubled opponent
  contract and Pass would end the auction, the exact DD ``delta`` if it doubled
  and everyone then passed drives a two-action loss
  ``-(p2(X) * delta) - cf_entropy * H(p2)`` with ``p2`` renormalized over
  {Pass, X}. ``nonfinal`` extends it to every legal-Double decision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..bridge.calls import DOUBLE, PASS
from ..contract.data import TorchDeals
from .model import policy_log_probs
from .rollout import FourSeatTrajectories
from .state import double_delta


def double_trajectory_losses(actor, critic, deals: TorchDeals, traj: FourSeatTrajectories,
                             entropy_weight: float = 0.01, policy_temperature: float = 1.0,
                             cf_entropy: float = 0.01, cf_nonfinal: bool = False,
                             tau: float = 0.5, cf_final_only: bool = False,
                             policy_cf: bool = True) -> dict:
    """``p2`` counterfactual for plain D5OWN4X nets; value mode for D5OWN4XV nets.

    Value mode: ``double_value`` MSE to the exact delta at every legal-X spot
    (or only Pass-ends spots with ``cf_final_only``) and a two-action cross-entropy
    of {Pass, X} toward ``sigmoid(double_value.detach() / tau)``, whose gradient
    ``p_X - w_X`` does not vanish at small ``p_X``. ``policy_cf=False`` trains the
    value only (warm-up).
    """
    if policy_temperature <= 0:
        raise ValueError("policy temperature must be positive")
    states = traj.states
    feats = states.features()
    seat = states.actor_seat
    hand = deals.hands[states.deal, seat]
    out = actor(hand, feats)
    legal = states.legal()
    logits = out["policy_logits"] / policy_temperature
    log_policy = policy_log_probs(out, legal, policy_temperature)
    policy = log_policy.exp()
    row = torch.arange(len(states), device=hand.device)
    chosen_logp = log_policy[row, traj.actions]
    entropy = -(policy * log_policy.masked_fill(~legal, 0.0)).sum(-1)

    pair = torch.stack((hand, deals.hands[states.deal, (seat + 2) % 4]), dim=1)
    value = critic(pair, feats)
    returns = traj.returns[traj.episode]
    standing = (traj.actions == DOUBLE) & (states.t == traj.standing_x[traj.row])
    credit = torch.where(standing, traj.final_delta[traj.row], torch.zeros_like(returns))
    action_return = returns + credit
    advantage = action_return - value.detach()
    groups = len(traj.score)
    policy_by_group = torch.zeros(groups, device=hand.device).scatter_add_(
        0, traj.episode, chosen_logp * advantage)
    entropy_by_group = torch.zeros(groups, device=hand.device).scatter_add_(
        0, traj.episode, entropy)

    extra: dict = {}
    value_mode = "double_value" in out
    can = legal[:, DOUBLE]
    final = can & (states.pass_count == 2)
    zero = logits.sum() * 0.0
    if value_mode:
        eligible = final if cf_final_only else can
        if bool(can.any()):
            delta_all = double_delta(states, deals)
            q = out["double_value"]
            gap_all = (out["double_gate"] / policy_temperature if "double_gate" in out
                       else logits[:, DOUBLE] - logits[:, PASS])
            target = torch.sigmoid(q.detach() / tau)
            value_loss = F.mse_loss(q[eligible], delta_all[eligible]) if bool(eligible.any()) else zero
            ce = F.binary_cross_entropy_with_logits(gap_all[eligible], target[eligible]) \
                if bool(eligible.any()) and policy_cf else zero
            cf_loss = value_loss + ce
            for name, mask in (("all", can), ("final", final)):
                if bool(mask.any()):
                    extra[f"double_value_mse_{name}"] = F.mse_loss(q[mask], delta_all[mask]).detach()
                    extra[f"double_profitable_share_{name}"] = (delta_all[mask] > 0).float().mean()
                    extra[f"double_p2_{name}"] = torch.sigmoid(gap_all[mask]).mean().detach()
                    extra[f"double_decisions_{name}"] = mask.sum().float()
            extra["double_value_mean"] = q[can].mean().detach()
            extra["double_delta_mean"] = delta_all[can].mean()
            extra["double_target_mean"] = target[eligible].mean() if bool(eligible.any()) else zero
            extra["double_ce"] = ce
            extra["double_value_loss"] = value_loss
        else:
            cf_loss = zero
            extra.update(double_value_loss=zero, double_ce=zero)
        cf_gain = cf_p = cf_profitable = torch.zeros((), device=hand.device)
    else:
        eligible = can if cf_nonfinal else final
    if value_mode:
        pass
    elif bool(eligible.any()):
        delta_cf = double_delta(states.subset(eligible.nonzero().squeeze(1)), deals)
        gap = logits[eligible, DOUBLE] - logits[eligible, PASS]
        p_x = torch.sigmoid(gap)
        h2 = F.softplus(gap) - p_x * gap     # binary entropy of sigmoid(gap)
        cf_loss = (-(p_x * delta_cf) - cf_entropy * h2).mean()
        cf_gain = (p_x * delta_cf).mean().detach()
        cf_p = p_x.mean().detach()
        cf_profitable = (delta_cf > 0).float().mean()
    else:
        cf_loss = logits.sum() * 0.0
        cf_gain = cf_p = cf_profitable = torch.zeros((), device=hand.device)

    rel = deals.rel_tricks(states.deal, seat)
    doubles = traj.actions == DOUBLE
    return {
        "policy_loss": -policy_by_group.mean(),
        "entropy": entropy_by_group.mean(),
        "policy_objective": -policy_by_group.mean() - entropy_weight * entropy_by_group.mean(),
        "critic_loss": F.mse_loss(value, returns),
        "q_loss": F.mse_loss(out["contract_q"][row, traj.actions], action_return),
        "trick_nll": F.cross_entropy(out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
        "double_cf_loss": cf_loss,
        "double_cf_decisions": eligible.sum().float(),
        "double_cf_gain": cf_gain,
        "double_cf_p2": cf_p,
        "double_cf_profitable_share": cf_profitable,
        "double_credit_mean": credit[doubles].mean() if bool(doubles.any())
        else torch.zeros((), device=hand.device),
        **extra,
        "return_mean": traj.returns.mean(),
        "advantage_mean": advantage.mean(),
        "advantage_std": advantage.std(),
    }
