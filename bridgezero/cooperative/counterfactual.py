"""All-legal-call continuation targets for cooperative auction learning.

DDS is used only to label training branches.  The deployable policy target is
formed from the frozen target network's conditional predictions, never from a
per-deal best-call label.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..contract.prefixes import exact_endpoint, expected_endpoint, net_outputs
from ..contract.targets import TARGET_SCALE, TorchScorer
from ..contract.train_auction import continuation_values


def supported_policy_target(values: torch.Tensor, legal: torch.Tensor,
                            temperature: float, support_floor: float) -> torch.Tensor:
    """Soft value policy mixed with uniform legal mass.

    ``support_floor`` is total uniform-mixture mass, rather than a per-action
    minimum.  This keeps the distribution normalized for auctions with different
    numbers of legal bids.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= support_floor < 1.0:
        raise ValueError("support_floor must be in [0, 1)")
    if values.shape != legal.shape:
        raise ValueError("values and legal mask shapes differ")
    soft = torch.softmax((values / temperature).masked_fill(~legal, -torch.inf), -1)
    uniform = legal.float() / legal.sum(-1, keepdim=True).clamp(min=1)
    return (1.0 - support_floor) * soft + support_floor * uniform


def counterfactual_losses(actor, target, deals, roots, scorer: TorchScorer,
                          temperature: float = 0.5,
                          support_floor: float = 0.02) -> dict[str, torch.Tensor]:
    """Train continuation residuals for every legal call at sampled states.

    For each legal root call, a frozen policy finishes the real auction and DDS
    supplies its score.  The residual target is continuation score minus the
    score from ending immediately at that call.  The policy is distilled from
    the target network's conditional endpoint prediction plus its learned
    residual, so hidden DDS labels never enter the policy target directly.
    """
    legal = roots.legal()
    with torch.no_grad():
        continuation = continuation_values(target, deals, roots, scorer, "policy")
        endpoint, _, rel = exact_endpoint(roots, deals, scorer)
        residual_target = (continuation - endpoint) / TARGET_SCALE
        target_out = net_outputs(target, deals, roots)
        endpoint_prediction = expected_endpoint(
            torch.softmax(target_out["trick_logits"], -1), roots, scorer) / TARGET_SCALE
        total_prediction = endpoint_prediction + target_out["continuation_residual"]
        policy_target = supported_policy_target(
            total_prediction, legal, temperature, support_floor)

    out = net_outputs(actor, deals, roots)
    log_policy = F.log_softmax(out["policy_logits"].masked_fill(~legal, -1e9), -1)
    policy = log_policy.exp()
    target_best = total_prediction.masked_fill(~legal, -torch.inf).argmax(-1)
    row = torch.arange(len(roots), device=legal.device)
    residual_error = (out["continuation_residual"] - residual_target) * TARGET_SCALE
    return {
        "policy_loss": -(policy_target * log_policy).sum(-1).mean(),
        # MSE learns the conditional mean required for expected-score bidding.
        # Huber instead trends toward a median on these skewed bridge scores and
        # previously produced a large Pass calibration error in the plain Q head.
        "residual_loss": F.mse_loss(
            out["continuation_residual"][legal], residual_target[legal]),
        "trick_nll": F.cross_entropy(
            out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
        "policy_entropy_bits": -(policy * log_policy.masked_fill(~legal, 0.0)).sum(-1).mean()
                               / math.log(2),
        "policy_prob_target_best": policy[row, target_best].mean(),
        "target_prob_min_legal": policy_target[legal].min(),
        "residual_rmse_points": residual_error[legal].pow(2).mean().sqrt(),
        "residual_target_abs_points": residual_target[legal].abs().mean() * TARGET_SCALE,
        "legal_calls": legal.sum().float(),
    }
