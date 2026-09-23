import json
from pathlib import Path

import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import AuctionContractNet, load_checkpoint
from bridgezero.contract.prefixes import CoopBatch
from bridgezero.contract.targets import TorchScorer
from bridgezero.cooperative.actor_critic import (
    CentralCritic,
    collect_trajectories,
    trajectory_losses,
)
from bridgezero.cooperative.train import parse_args, run

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def test_full_vocabulary_trajectories_are_legal_and_terminal():
    data = deals()
    actor = AuctionContractNet(16, 8, 1)
    trajectories = collect_trajectories(
        actor, data, 32, torch.Generator().manual_seed(4), TorchScorer(), max_decisions=40)
    assert bool(trajectories.terminal.ended.all())
    assert trajectories.actions.min() >= 0 and trajectories.actions.max() < 36
    for state, action in zip(trajectories.states.call_lists(), trajectories.actions.tolist()):
        assert all(0 <= call < 36 for call in state)
        if action < 35 and state:
            bids = [call for call in state if call < 35]
            assert not bids or action > max(bids)


def test_terminal_return_reaches_every_policy_decision():
    data = deals()
    actor = AuctionContractNet(16, 8, 1)
    critic = CentralCritic(16, 8, 1)
    trajectories = collect_trajectories(
        actor, data, 16, torch.Generator().manual_seed(8), TorchScorer())
    losses = trajectory_losses(actor, critic, data, trajectories)
    actor.zero_grad(set_to_none=True)
    losses["policy_objective"].backward()
    assert actor.policy_head.weight.grad is not None
    assert float(actor.policy_head.weight.grad.norm()) > 0
    assert set(trajectories.episode.tolist()) == set(range(16))


def test_behavior_temperature_is_part_of_the_on_policy_loss():
    data = deals()
    actor = AuctionContractNet(16, 8, 1)
    critic = CentralCritic(16, 8, 1)
    trajectories = collect_trajectories(
        actor, data, 8, torch.Generator().manual_seed(12), TorchScorer(), temperature=0.7)
    # Zero-initialized policy heads are uniform, so perturb one logit before comparing.
    with torch.no_grad():
        actor.policy_head.bias[0] = 1.0
    cold = trajectory_losses(actor, critic, data, trajectories, policy_temperature=0.7)
    unit = trajectory_losses(actor, critic, data, trajectories, policy_temperature=1.0)
    assert not torch.allclose(cold["policy_loss"], unit["policy_loss"])


def test_bounded_logits_preserve_greedy_call_and_enforce_probability_floor():
    data = deals()
    actor = AuctionContractNet(16, 8, 1, policy_logit_bound=3.0)
    roots = CoopBatch.start(torch.tensor([0]), torch.tensor([0]),
                            torch.tensor([0]), torch.tensor([0]))
    with torch.no_grad():
        actor.policy_head.weight.zero_()
        actor.policy_head.bias.copy_(torch.linspace(-100.0, 100.0, 36))
        hidden = actor.encode(data.hands[roots.deal, roots.actor_seat], roots.features())
        raw = actor.policy_head(hidden)
        bounded = actor(data.hands[roots.deal, roots.actor_seat],
                        roots.features())["policy_logits"]
    assert bounded.abs().max() <= 3.0
    assert torch.equal(raw.argmax(-1), bounded.argmax(-1))
    probability = torch.softmax(bounded, -1)
    theoretical_floor = 1.0 / (1.0 + 35.0 * torch.exp(torch.tensor(6.0)))
    assert float(probability.min()) >= float(theoretical_floor) - 1e-8


def test_from_scratch_trainer_smoke(tmp_path):
    out = tmp_path / "scratch"
    args = parse_args([
        "--data", str(SMOKE), "--out", str(out),
        "--train-start", "0", "--train-count", "96",
        "--val-start", "96", "--val-count", "16",
        "--eval-start", "112", "--eval-count", "16",
        "--ground-steps", "2", "--pg-steps", "2", "--batch", "16",
        "--episodes", "16", "--width", "16", "--suit-width", "8", "--depth", "1",
        "--policy-logit-bound", "3", "--eval-every", "1", "--target-every", "1",
        "--threads", "1"])
    report = run(args)
    assert report["from_scratch"] and report["discount"] == 1.0
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert all(record["validation"]["policy"]["effective_calls_mean"] >= 1.0
               for record in records)
    net, meta = load_checkpoint(out / "best.pt", stage="D4PG")
    assert isinstance(net, AuctionContractNet) and meta["args"]["ground_steps"] == 2
    assert net.policy_logit_bound == 3.0 and net.config["policy_logit_bound"] == 3.0
    assert meta["args"]["ground_lr"] == 1e-3 and meta["args"]["pg_lr"] == 3e-4
    assert meta["args"]["ground_trick_weight"] == 1.0
    assert meta["args"]["ground_q_weight"] == 1.0
    assert (out / "last_state.pt").exists() and (out / "eval_rows.npz").exists()


def test_from_scratch_joint_search_trainer_smoke(tmp_path):
    out = tmp_path / "scratch_jps"
    args = parse_args([
        "--data", str(SMOKE), "--out", str(out),
        "--train-start", "0", "--train-count", "96",
        "--val-start", "96", "--val-count", "16",
        "--eval-start", "112", "--eval-count", "16",
        "--ground-steps", "1", "--pg-steps", "0",
        "--jps-stop-steps", "1", "--jps-full-steps", "1",
        "--jps-roots", "2", "--jps-inner-steps", "1",
        "--batch", "16", "--width", "16", "--suit-width", "8", "--depth", "1",
        "--eval-every", "1", "--jps-eval-every", "1", "--threads", "1"])
    report = run(args)
    assert report["from_scratch"]
    net, meta = load_checkpoint(out / "best.pt", stage="D4JPS")
    assert isinstance(net, AuctionContractNet)
    assert meta["args"]["pg_steps"] == 0
    assert meta["args"]["jps_stop_steps"] == 1
    phases = [json.loads(line)["phase"]
              for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert "joint_stop" in phases and "joint_policy" in phases
