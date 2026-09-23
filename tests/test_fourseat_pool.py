from pathlib import Path

import torch

from bridgezero.bridge.calls import DOUBLE
from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import AuctionContractNet
from bridgezero.contract.targets import TorchScorer
from bridgezero.fourseat.model import (
    FourSeatCritic,
    FourSeatDoubleGateNet,
    FourSeatNet,
    load_fourseat_checkpoint,
    save_fourseat_checkpoint,
    warm_start_from_fourseat,
)
from bridgezero.fourseat.rollout import collect_trajectories, fourseat_validation
from bridgezero.fourseat.train import parse_args, run

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def fourseat_checkpoint(tmp_path, width=24):
    torch.manual_seed(1)
    net, critic = FourSeatNet(width, 8, 2), FourSeatCritic(width, 8, 2)
    with torch.no_grad():
        for p in (*net.parameters(), *critic.parameters()):
            p.normal_(0, 0.3)
    path = tmp_path / "e18" / "best.pt"
    save_fourseat_checkpoint(path, net, critic, step=7)
    return path, net


def test_warm_start_from_fourseat_plays_identically(tmp_path):
    path, source = fourseat_checkpoint(tmp_path)
    actor, _, meta = warm_start_from_fourseat(path, double_bias=-6.0)
    assert isinstance(actor, FourSeatDoubleGateNet) and meta["init_step"] == 7
    data, scorer = deals(), TorchScorer()
    before = fourseat_validation(source, data, scorer)
    after = fourseat_validation(actor, data, scorer, doubles=True)
    assert after["doubles"] == 0
    for key in ("own_score", "own_ns", "own_ew", "calls", "passout", "both_sides_bid"):
        assert after[key] == before[key], key


def test_pool_episodes_train_only_the_learner_side(tmp_path):
    path, source = fourseat_checkpoint(tmp_path)
    actor, _, _ = warm_start_from_fourseat(path)
    with torch.no_grad():
        actor.double_gate_head.bias.fill_(0.0)
    blind = AuctionContractNet(24, 8, 2)
    pool = {1: (source, 0.3), 2: (blind, 0.3)}
    traj = collect_trajectories(actor, deals(), 96, torch.Generator().manual_seed(3), TorchScorer(),
                                silent_frac=0.5, doubles=True, pool=pool)
    opp = traj.opponent
    assert set(opp.tolist()) == {0, 1, 2}
    assert bool((traj.terminal.silent[opp > 0] < 0).all())
    frozen_side = traj.frozen_side[traj.row]
    assert not bool((traj.states.side == frozen_side).any())
    # frozen players never double
    terminal = traj.terminal
    for row in (opp > 0).nonzero().squeeze(1).tolist():
        for pos, call in enumerate(terminal.history[row].tolist()):
            if call == DOUBLE:
                assert (int(terminal.dealer[row]) + pos) % 2 != int(traj.frozen_side[row])
    m = fourseat_validation(actor, deals(), TorchScorer(), doubles=True, frozen_net=blind)
    assert 0.0 <= m["double_rate"] <= 1.0 and "learner_table_score" in m


def test_pool_trainer_smoke(tmp_path):
    path, _ = fourseat_checkpoint(tmp_path, width=16)
    blind_path = tmp_path / "blind.pt"
    from bridgezero.contract.model import save_checkpoint
    from bridgezero.cooperative.actor_critic import CentralCritic
    save_checkpoint(blind_path, AuctionContractNet(16, 8, 1), "D4PG", step=0,
                    critic_config=CentralCritic(16, 8, 1).config,
                    critic=CentralCritic(16, 8, 1).state_dict())
    out = tmp_path / "e20"
    report = run(parse_args([
        "--data", str(SMOKE), "--out", str(out), "--init", str(path), "--init-fourseat",
        "--double-tau", "0.1", "--pool", f"E18=four:{path}:0.25",
        "--pool", f"E15d=zero:{blind_path}:0.25",
        "--train-start", "0", "--train-count", "96", "--val-start", "96", "--val-count", "16",
        "--eval-start", "112", "--eval-count", "16", "--steps", "2", "--episodes", "16",
        "--eval-every", "1", "--threads", "1", "--max-double-rate", "1.1"]))
    assert set(report["fourseat_pool"]) == {"E18", "E15d"}
    net, meta = load_fourseat_checkpoint(out / "best.pt")
    assert meta["stage"] == "D5OWN4XD"


def test_long_run_blocks_snapshots_and_resume(tmp_path):
    path, _ = fourseat_checkpoint(tmp_path, width=16)
    out = tmp_path / "e21"
    common = [
        "--data", str(SMOKE), "--out", str(out), "--init", str(path), "--init-fourseat",
        "--double-tau", "0.1", "--pool", f"E18=four:{path}:0.1",
        "--lr-schedule", "constant", "--train-block-every", "2", "--train-block-size", "40",
        "--train-pool-start", "0", "--train-pool-end", "96",
        "--val-start", "96", "--val-count", "16", "--eval-start", "112", "--eval-count", "16",
        "--episodes", "8", "--eval-every", "2", "--state-every", "2", "--snapshot-every", "2",
        "--threads", "1", "--max-double-rate", "1.1"]
    run(parse_args(common + ["--steps", "2"]))
    first = torch.load(out / "last_state.pt", weights_only=False)
    assert first["step"] == 2 and (out / "ckpt_step2.pt").exists()
    run(parse_args(common + ["--steps", "4", "--resume"]))
    second = torch.load(out / "last_state.pt", weights_only=False)
    assert second["step"] == 4 and (out / "ckpt_step4.pt").exists()
    import json
    steps = [json.loads(line)["step"] for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert steps[0] == 0 and 4 in steps and steps.count(0) == 1
    starts = {json.loads(line).get("train_block_start")
              for line in (out / "train_log.jsonl").read_text().splitlines()} - {None}
    assert all(0 <= s <= 56 for s in starts)
