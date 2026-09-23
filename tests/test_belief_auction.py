from pathlib import Path

import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import (
    AuctionContractNet,
    BeliefAuctionNet,
    ResidualBeliefAuctionNet,
    load_checkpoint,
    objective_for,
    save_checkpoint,
)
from bridgezero.contract.evaluate_auction import auction_rows, continue_auctions, trajectory_belief_metrics
from bridgezero.contract.prefixes import CoopBatch, sample_prefixes
from bridgezero.contract.targets import TorchScorer
from bridgezero.contract.train_auction import d1_losses, parse_args

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    owners, tricks = load_dataset(SMOKE)
    return TorchDeals(owners, tricks)


def test_belief_auction_shapes_mask_own_cards_and_checkpoint(tmp_path):
    net = BeliefAuctionNet(32, 8, 1, 16)
    hand = deals().hands[:7, 0]
    auction = torch.zeros(7, 77)
    out = net(hand, auction)
    assert out["partner_belief_logits"].shape == (7, 52)
    assert out["trick_logits"].shape == (7, 2, 5, 14)
    assert out["contract_q"].shape == out["policy_logits"].shape == (7, 36)
    assert torch.equal(out["partner_probability"][hand.bool()], torch.zeros(7 * 13))
    path = tmp_path / "belief.pt"
    save_checkpoint(path, net, "D4", "rollout")
    loaded, meta = load_checkpoint(path, stage="D4")
    assert isinstance(loaded, BeliefAuctionNet)
    assert meta["objective"] == objective_for("D4")


def test_value_path_consumes_differentiable_partner_belief():
    data = deals()
    net = BeliefAuctionNet(32, 8, 1, 16)
    batch, _ = sample_prefixes(data, 32, torch.Generator().manual_seed(8), max_depth=3)
    losses = d1_losses(net, net, data, batch, TorchScorer(), 0.5,
                       policy_source="rollout")
    assert {"belief_loss", "belief_count_loss"} <= set(losses)
    net.zero_grad(set_to_none=True)
    losses["trick_nll"].backward()
    assert net.belief_head.weight.grad is not None
    assert float(net.belief_head.weight.grad.abs().sum()) > 0


def test_full_auction_stage_has_explicit_safe_recipe():
    args = parse_args(["--data", str(SMOKE), "--out", "unused", "--stage", "D4",
                       "--architecture", "residual_belief", "--continuation-frac", "1",
                       "--continuation-rule", "policy", "--endpoint-warmup-steps", "10"])
    assert args.policy_source == ""  # filled with the D4 default by the runner
    assert objective_for("D4") == objective_for("D4", "rollout")


def test_residual_belief_model_starts_bit_exact_to_base_policy():
    torch.manual_seed(19)
    base = AuctionContractNet(32, 8, 1)
    residual = ResidualBeliefAuctionNet(32, 8, 1, 16)
    residual.load_base(base)
    hand = deals().hands[:11, 0]
    auction = torch.randn(11, 77)
    expected, actual = base(hand, auction), residual(hand, auction)
    for key in ("trick_logits", "contract_q", "policy_logits"):
        assert torch.equal(expected[key], actual[key])
    residual.freeze_base()
    frozen = {name for name, parameter in residual.named_parameters()
              if not parameter.requires_grad}
    assert "policy_head.weight" in frozen and "belief_head.weight" not in frozen


def test_on_policy_belief_metric_tracks_every_decision_depth():
    data = deals().head(4)
    net = BeliefAuctionNet(32, 8, 1, 16).eval()
    rows = auction_rows(data.n)
    states = []
    continue_auctions(net, data, CoopBatch.start(*rows), TorchScorer(), "q", record=states)
    metrics = trajectory_belief_metrics(net, data, states)
    assert set(metrics) == {"intact", "partner_masked"}
    assert set(metrics["intact"]) == set(metrics["partner_masked"])
    assert metrics["intact"]["0"]["rows"] == len(rows[0])
    assert metrics["intact"]["0"] == metrics["partner_masked"]["0"]
