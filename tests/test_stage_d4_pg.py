"""Stage D4 fixes: MSE Q regression, exact enumerated policy value, listener signal credit."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import (
    AuctionContractNet,
    ResidualBeliefAuctionNet,
    load_checkpoint,
    objective_for,
    save_checkpoint,
)
from bridgezero.contract.prefixes import sample_prefixes
from bridgezero.contract.targets import TorchScorer
from bridgezero.contract.train_auction import d1_losses, parse_args, run, signal_nats

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"
SMALL = ["--data", str(SMOKE), "--train-count", "96", "--val-start", "96", "--val-count", "16",
         "--eval-start", "-16", "--eval-count", "16", "--steps", "4", "--batch", "32",
         "--eval-every", "2", "--target-every", "2", "--width", "16", "--suit-width", "8",
         "--depth", "1", "--threads", "1", "--val-q-prefixes", "32"]


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def freeze(run_dir: Path) -> None:
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
             for p in sorted(run_dir.iterdir()) if p.is_file()]
    (run_dir / "FROZEN.sha256").write_text("\n".join(lines) + "\n")


def test_objective_names_q_loss_and_exact_pg_and_checkpoint_round_trips(tmp_path):
    assert objective_for("D2") == objective_for("D2", "q", "huber")
    assert "mse(legal_q_36" in objective_for("D2", "q", "mse")
    assert "huber(" not in objective_for("D4", "exact_pg", "mse")
    assert "policy_value(" in objective_for("D4", "exact_pg")
    assert "policy_ce(" not in objective_for("D4", "exact_pg")
    with pytest.raises(ValueError):
        objective_for("D2", "exact_pg")
    with pytest.raises(ValueError):
        objective_for("D2", "q", "l1")
    net = AuctionContractNet(16, 8, 1)
    save_checkpoint(tmp_path / "m.pt", net, "D4", "exact_pg", "mse")
    _, meta = load_checkpoint(tmp_path / "m.pt", stage="D4")
    assert meta["q_loss"] == "mse" and meta["policy_source"] == "exact_pg"
    save_checkpoint(tmp_path / "h.pt", net, "D2", "q")
    raw = torch.load(tmp_path / "h.pt", weights_only=False)
    assert "q_loss" not in raw                      # legacy layout for Huber checkpoints
    assert load_checkpoint(tmp_path / "h.pt")[1]["q_loss"] == "huber"


def test_exact_pg_moves_policy_toward_higher_value_and_lower_signal_cost():
    data = deals()
    torch.manual_seed(3)
    net = AuctionContractNet(16, 8, 1)
    batch, _ = sample_prefixes(data, 1, torch.Generator().manual_seed(4), opening_prob=1.0)
    legal = batch.legal()
    calls = legal[0].nonzero().squeeze(1)
    good, bad = int(calls[0]), int(calls[1])
    values = torch.full((1, 36), -5.0)
    values[0, good] = 2.0

    def grad_on(signal=None, weight=0.0, cont=values):
        net.zero_grad(set_to_none=True)
        losses = d1_losses(net, net, data, batch, TorchScorer(), 0.5,
                           continuation=(torch.ones(1, dtype=torch.bool), cont * 100),
                           policy_source="exact_pg", signal=signal, signal_weight=weight)
        losses["policy_loss"].backward()
        return net.policy_head.bias.grad.clone()

    # Gradient descent raises a logit whose gradient is negative.
    grad = grad_on()
    assert grad[good] < 0 and grad[bad] > 0
    flat = torch.zeros(1, 36)
    signal = torch.zeros(1, 36)
    signal[0, good] = 30.0
    assert grad_on(signal, 1.0, flat)[good] > 0     # an uninformative call is discouraged
    stats: dict = {}
    d1_losses(net, net, data, batch, TorchScorer(), 0.5, policy_source="exact_pg",
              q_loss="mse", stats=stats)
    assert {"q_rmse_points", "q_pass_bias_points", "policy_entropy_bits"} <= set(stats)


def test_signal_nats_scores_every_legal_call_with_the_speakers_hand():
    data = deals()
    net = ResidualBeliefAuctionNet(16, 8, 1, 8).eval()
    batch, _ = sample_prefixes(data, 20, torch.Generator().manual_seed(5), max_depth=3)
    nats = signal_nats(net, data, batch)
    legal = batch.legal()
    assert nats.shape == (20, 36)
    assert bool((nats[~legal] == 0).all()) and bool((nats[legal] > 0).all())


def test_terminal_pass_is_neutral_in_signal_auxiliary():
    data = deals()
    net = ResidualBeliefAuctionNet(16, 8, 1, 8).eval()
    batch, _ = sample_prefixes(data, 24, torch.Generator().manual_seed(17),
                               opening_prob=0.0, max_depth=3)
    ending = batch.pass_ends()
    assert bool(ending.any())
    nats = signal_nats(net, data, batch)
    legal_bids = batch.legal()[:, :-1]
    mean_bid = ((nats[:, :-1] * legal_bids).sum(-1)
                / legal_bids.sum(-1).clamp(min=1))
    rows = ending & legal_bids.any(-1)
    assert torch.allclose(nats[rows, -1], mean_bid[rows])


def test_d4_exact_pg_trainer_logs_key_metrics_and_checks_init_lineage(tmp_path):
    d1 = tmp_path / "d1"
    run(parse_args([*SMALL, "--out", str(d1)]))
    freeze(d1)
    d4_args = [*SMALL, "--stage", "D4", "--architecture", "residual_belief",
               "--init", str(d1 / "best.pt"), "--freeze-base", "--continuation-frac", "1",
               "--endpoint-warmup-steps", "1", "--belief-pretrain-steps", "1",
               "--policy-source", "exact_pg", "--q-loss", "mse", "--signal-weight", "0.1"]
    d4 = tmp_path / "d4"
    report = run(parse_args([*d4_args, "--out", str(d4)]))
    log = [json.loads(line) for line in (d4 / "train_log.jsonl").read_text().splitlines()]
    assert log[0]["before_update"] and log[0]["val_key"]["imps_vs_start"] == 0
    assert log[1]["phase"] == "belief_pretrain" and "q_loss" not in log[1]
    joint = log[-1]
    assert joint["phase"] == "joint" and "signal_nats_under_policy" in joint["train_batch"]
    for name in ("score", "gap_to_dd_par", "imp_gap_to_dd_par", "imps_vs_start", "make_rate"):
        assert name in joint["val_key"]
    assert {"endpoint", "continuation"} <= set(joint["val_q_fit"])
    assert "top13_recall" in joint["val_belief"]["intact"]
    assert "belief_trajectory_policy" in report     # residual belief nets are recognized
    assert load_checkpoint(d4 / "best.pt")[1]["q_loss"] == "mse"
    with pytest.raises(ValueError):                 # signal credit needs exact_pg
        run(parse_args([*d4_args[:-6], "--out", str(tmp_path / "x"), "--signal-weight", "0.1"]))
    with pytest.raises(ValueError, match="init lineage"):   # init trained on this val range
        run(parse_args([*SMALL, "--out", str(tmp_path / "y"), "--stage", "D2",
                        "--init", str(d1 / "best.pt"), "--train-start", "16",
                        "--train-count", "80", "--val-start", "0"]))
