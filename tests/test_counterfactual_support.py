import json
from pathlib import Path

import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import ContinuationResidualAuctionNet, load_checkpoint
from bridgezero.contract.prefixes import CoopBatch
from bridgezero.contract.targets import TorchScorer
from bridgezero.cooperative.actor_critic import mix_legal_uniform
from bridgezero.cooperative.counterfactual import (
    counterfactual_losses,
    supported_policy_target,
)
from bridgezero.cooperative.train import parse_args, run

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def opening_roots(n=2):
    return CoopBatch.start(torch.arange(n), torch.zeros(n, dtype=torch.long),
                           torch.zeros(n, dtype=torch.long),
                           torch.zeros(n, dtype=torch.long))


def test_uniform_mixtures_preserve_legality_and_give_every_legal_call_support():
    legal = torch.tensor([[True, True, False], [False, True, True]])
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    mixed = mix_legal_uniform(probs, legal, 0.2)
    target = supported_policy_target(torch.tensor([[10.0, -10.0, 20.0],
                                                    [30.0, -5.0, 5.0]]),
                                     legal, 1.0, 0.2)
    for distribution in (mixed, target):
        assert torch.allclose(distribution.sum(-1), torch.ones(2))
        assert bool((distribution[legal] >= 0.1).all())
        assert bool((distribution[~legal] == 0).all())


def test_residual_head_starts_at_exactly_zero():
    net = ContinuationResidualAuctionNet(16, 8, 1)
    roots = opening_roots()
    out = net(deals().hands[roots.deal, roots.actor_seat], roots.features())
    assert torch.equal(out["continuation_residual"], torch.zeros(2, 36))


def test_counterfactual_ce_recovers_an_action_with_tiny_actor_probability():
    data = deals()
    actor = ContinuationResidualAuctionNet(16, 8, 1)
    target = ContinuationResidualAuctionNet(16, 8, 1)
    target.load_state_dict(actor.state_dict())
    good = 0
    with torch.no_grad():
        actor.policy_head.bias[good] = -40.0
        target.continuation_head.bias[good] = 20.0
    losses = counterfactual_losses(
        actor, target, data, opening_roots(1), TorchScorer(),
        temperature=0.5, support_floor=0.02)
    actor.zero_grad(set_to_none=True)
    losses["policy_loss"].backward()
    assert float(actor.policy_head.bias.grad[good]) < -0.9


def test_pass_continuation_residual_is_zero_when_pass_ends_auction():
    data = deals()
    net = ContinuationResidualAuctionNet(16, 8, 1)
    roots = opening_roots(1)
    roots.apply(torch.tensor([0]), torch.tensor([True]))
    losses = counterfactual_losses(net, net, data, roots, TorchScorer())
    assert torch.isfinite(losses["residual_loss"])
    # A terminal Pass has no continuation beyond the standing endpoint.  The
    # all-zero residual prediction is therefore already exact in this slot.
    net.zero_grad(set_to_none=True)
    losses["residual_loss"].backward()
    assert abs(float(net.continuation_head.bias.grad[-1])) < 1e-7


def test_from_scratch_counterfactual_trainer_smoke(tmp_path):
    out = tmp_path / "scratch_cf"
    args = parse_args([
        "--data", str(SMOKE), "--out", str(out),
        "--train-start", "0", "--train-count", "96",
        "--val-start", "96", "--val-count", "16",
        "--eval-start", "112", "--eval-count", "16",
        "--ground-steps", "1", "--pg-steps", "0", "--cf-steps", "1",
        "--cf-roots", "2", "--episodes", "4", "--batch", "16",
        "--width", "16", "--suit-width", "8", "--depth", "1",
        "--eval-every", "1", "--target-every", "1", "--threads", "1"])
    report = run(args)
    assert report["from_scratch"]
    net, meta = load_checkpoint(out / "last.pt", stage="D4CF")
    assert isinstance(net, ContinuationResidualAuctionNet)
    assert meta["args"]["cf_steps"] == 1
    phases = [json.loads(line)["phase"]
              for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert "all_action_cf" in phases
