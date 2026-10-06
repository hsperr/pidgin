"""The three train.sh stages end to end on tiny sizes, four-seat resume, training blocks."""

import json
from pathlib import Path

import torch

from training import ground
from training.contract.data import block_starts
from training.contract.model import AuctionContractNet, CentralCritic, save_checkpoint
from training.fourseat import train as fourseat
from training.fourseat.model import (
    FourSeatCompetitiveCritic,
    FourSeatCompetitiveNet,
    load_fourseat_checkpoint,
    warm_start_competitive,
)

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"
# pool 0..96 in 32-deal blocks; validation 96..112; held-out eval the last 16
DATA = ["--data", str(SMOKE), "--train-pool-start", "0", "--train-pool-end", "96",
        "--train-block-size", "32", "--val-start", "96", "--val-count", "16",
        "--eval-start", "-16", "--eval-count", "16", "--seed", "1", "--threads", "1"]
NET = ["--width", "16", "--suit-width", "8", "--depth", "1"]
GUARDS = ["--max-double-rate", "1.1", "--max-sac-rate", "10", "--max-level5-rise", "1"]


def log_steps(out: Path) -> list[int]:
    return [json.loads(line)["step"] for line in (out / "train_log.jsonl").read_text().splitlines()]


def test_block_starts_cover_the_pool_without_overlap():
    blocks = block_starts(100, 1100, 100, seed=3)
    assert sorted(blocks) == list(range(100, 1100, 100))
    assert blocks != sorted(blocks)                              # permuted
    assert blocks == block_starts(100, 1100, 100, seed=3)        # seeded
    assert blocks != block_starts(100, 1100, 100, seed=4)
    # a partial last block is dropped, never overlapped
    ragged = sorted(block_starts(0, 250, 100, seed=0))
    assert ragged == [0, 100]


def test_three_stage_pipeline(tmp_path):
    g = tmp_path / "1_ground"
    result = ground.run(ground.parser().parse_args([
        *DATA, *NET, "--out", str(g), "--train-block-every", "10", "--batch", "32",
        "--eval-every", "1", "--patience", "3", "--warmup", "1"]))
    assert (g / "best.pt").exists() and (g / "result.json").exists()
    assert result["last_step"] == result["best_step"] + 3 < 30      # early stop, not pool end
    assert log_steps(g)[-1] == result["last_step"]

    own = tmp_path / "2_own"
    report = fourseat.run(fourseat.parse_args([
        *DATA, *GUARDS, "--out", str(own), "--init", str(g / "best.pt"),
        "--select", "own", "--table-weight", "0", "--train-block-every", "4",
        "--episodes", "16", "--eval-every", "1", "--patience", "2", "--snapshot-every", "2"]))
    assert (own / "best.pt").exists() and (own / "eval.json").exists()
    assert report["early_stop"].startswith("no new best own") and report["last_step"] < 12
    assert report["eval_range"] == [112, 128]
    net, meta = load_fourseat_checkpoint(own / "best.pt")
    assert meta["stage"] == "D5OWN4XC" and isinstance(net, FourSeatCompetitiveNet)
    assert net.redouble and net.sacrifice and meta["init_path"] == str(g / "best.pt")

    table = tmp_path / "3_table"
    report = fourseat.run(fourseat.parse_args([
        *DATA, *GUARDS, "--out", str(table), "--init", str(own / "last.pt"),
        "--select", "imp", "--imp-opponent", str(own / "last.pt"), "--table-weight", "1.0",
        "--league-frac", "0.5", "--league-every", "1", "--any-seat-double", "--gate-pg",
        "--train-block-every", "2", "--steps", "3", "--episodes", "16",
        "--eval-every", "2", "--snapshot-every", "2"]))
    assert (table / "best.pt").exists() and (table / "eval.json").exists()
    assert report["select"] == "imp" and report["last_step"] == 3
    records = [json.loads(line) for line in (table / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 2, 3]            # no step-1 match
    _, best_meta = load_fourseat_checkpoint(table / "best.pt")
    assert best_meta["step"] >= 0                               # the parent is a candidate
    assert all("imps_vs_opponent" in r["validation"]["fourseat"] for r in records)
    assert (table / "matches" / "step2" / "results.json").exists()
    assert records[-1]["league_size"] == 3                      # snapshots 0, 1, 2
    _, meta = load_fourseat_checkpoint(table / "best.pt")
    assert meta["any_seat_double"] is True
    # a D5OWN4XC checkpoint continues unchanged
    actor, critic, _ = warm_start_competitive(table / "last.pt")
    last, _ = load_fourseat_checkpoint(table / "last.pt")
    assert isinstance(critic, FourSeatCompetitiveCritic)
    for key, value in last.state_dict().items():
        assert torch.equal(actor.state_dict()[key], value), key


def test_fourseat_resume_continues_steps_and_league(tmp_path):
    torch.manual_seed(0)
    init = tmp_path / "init" / "best.pt"
    save_checkpoint(init, AuctionContractNet(16, 8, 1), "D4PG", step=0,
                    critic_config=CentralCritic(16, 8, 1).config,
                    critic=CentralCritic(16, 8, 1).state_dict())
    out = tmp_path / "four"
    common = [*DATA, *GUARDS, "--out", str(out), "--init", str(init), "--episodes", "8",
              "--train-block-every", "2", "--eval-every", "1", "--snapshot-every", "2",
              "--league-frac", "0.5", "--league-every", "1", "--patience", "100"]
    fourseat.run(fourseat.parse_args(common + ["--steps", "2"]))
    first = torch.load(out / "last_state.pt", weights_only=False)
    assert first["step"] == 2 and (out / "ckpt_step2.pt").exists()
    assert [s for s, _ in first["league_snapshots"]] == [0, 1, 2]

    fourseat.run(fourseat.parse_args(common + ["--steps", "4", "--resume"]))
    second = torch.load(out / "last_state.pt", weights_only=False)
    assert second["step"] == 4 and (out / "ckpt_step4.pt").exists()
    assert (out / "run_resume_step2.json").exists()
    assert [s for s, _ in second["league_snapshots"]] == [0, 1, 2, 3, 4]
    for (_, a), (_, b) in zip(first["league_snapshots"], second["league_snapshots"]):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert log_steps(out) == [0, 1, 2, 3, 4]
    records = [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]
    assert all(r["train_block_start"] in (0, 32, 64) for r in records[1:])
    assert records[3]["league_size"] == 3                       # restored, not restarted at 1


def test_imp_stage_keeps_parent_when_every_update_trips_guard(tmp_path, monkeypatch):
    init = tmp_path / "init" / "best.pt"
    save_checkpoint(init, AuctionContractNet(16, 8, 1), "D4PG", step=0,
                    critic_config=CentralCritic(16, 8, 1).config,
                    critic=CentralCritic(16, 8, 1).state_dict())
    monkeypatch.setattr(fourseat, "match_imps", lambda args, out, step, ckpt: 0.0 if step == 0 else -0.5)
    out = tmp_path / "table"
    report = fourseat.run(fourseat.parse_args([
        *DATA, "--out", str(out), "--init", str(init), "--select", "imp",
        "--imp-opponent", str(init), "--table-weight", "1", "--episodes", "8",
        "--train-block-every", "2", "--steps", "1", "--eval-every", "1",
        "--max-double-rate", "-1", "--max-level5-rise", "1",
    ]))
    _, best = load_fourseat_checkpoint(out / "best.pt")
    assert best["step"] == 0
    assert report["select_metric"] == 0.0
    assert report["early_stop"].startswith("double rate")


def test_imp_stage_measures_nonparent_starting_score(tmp_path, monkeypatch):
    init = tmp_path / "init" / "best.pt"
    save_checkpoint(init, AuctionContractNet(16, 8, 1), "D4PG", step=0,
                    critic_config=CentralCritic(16, 8, 1).config,
                    critic=CentralCritic(16, 8, 1).state_dict())
    seen = []

    def match(args, out, step, checkpoint):
        seen.append(step)
        return 0.25 if step == 0 else -0.5

    monkeypatch.setattr(fourseat, "match_imps", match)
    out = tmp_path / "table"
    report = fourseat.run(fourseat.parse_args([
        *DATA, "--out", str(out), "--init", str(init), "--select", "imp",
        "--imp-opponent", str(tmp_path / "different_opponent.pt"),
        "--table-weight", "1", "--episodes", "8", "--train-block-every", "2",
        "--steps", "1", "--eval-every", "1", "--max-double-rate", "-1",
    ]))
    assert seen == [0, 1]
    assert report["select_metric"] == 0.25
    assert load_fourseat_checkpoint(out / "best.pt")[1]["step"] == 0


def test_zero_patience_completes_the_training_blocks(tmp_path):
    init = tmp_path / "init" / "best.pt"
    save_checkpoint(init, AuctionContractNet(16, 8, 1), "D4PG", step=0,
                    critic_config=CentralCritic(16, 8, 1).config,
                    critic=CentralCritic(16, 8, 1).state_dict())
    out = tmp_path / "own"
    report = fourseat.run(fourseat.parse_args([
        *DATA, *GUARDS, "--out", str(out), "--init", str(init),
        "--episodes", "8", "--train-block-every", "2", "--eval-every", "1",
        "--patience", "0", "--snapshot-every", "2",
    ]))
    assert report["last_step"] == 6  # three blocks, two steps per block
    assert report["early_stop"] is None


def test_grounding_checks_final_step_and_zero_patience(tmp_path):
    out = tmp_path / "ground"
    result = ground.run(ground.parser().parse_args([
        *DATA, *NET, "--out", str(out), "--train-block-every", "2", "--batch", "8",
        "--max-steps", "3", "--eval-every", "2", "--patience", "0"]))
    assert result["last_step"] == 3
    assert log_steps(out) == [0, 2, 3]


def test_invalid_training_pools_fail_before_training(tmp_path):
    import pytest
    from training.contract.data import validate_training_pool
    for start, end, size, every in [(-1, 96, 32, 2), (0, 129, 32, 2),
                                    (0, 96, 0, 2), (0, 16, 32, 2), (0, 96, 32, 0)]:
        with pytest.raises(ValueError):
            validate_training_pool(128, start, end, size, every)


def test_public_script_trains_random_through_d_and_skips_finished_stages(tmp_path):
    import subprocess
    import sys
    import os
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "public"
    env = {**os.environ, "PYTHON": sys.executable}
    subprocess.run(["bash", str(root / "train.sh"), "--smoke", str(out)],
                   cwd=root, env=env, check=True, capture_output=True, text=True)
    own_simple = json.loads((out / "3_simple" / "run.json").read_text())
    table = json.loads((out / "4_D" / "run.json").read_text())
    assert own_simple["code_word_penalty"] == 0.2
    assert table["code_word_penalty"] == 0.2 and table["table_weight"] == 1
    assert table["imp_opponent"] == str(out / "3_simple" / "last.pt")
    assert (out / "4_D" / "best.pt").is_file()
    before = (out / "4_D" / "train_log.jsonl").read_bytes()
    subprocess.run(["bash", str(root / "train.sh"), "--smoke", str(out)],
                   cwd=root, env=env, check=True, capture_output=True, text=True)
    assert (out / "4_D" / "train_log.jsonl").read_bytes() == before
