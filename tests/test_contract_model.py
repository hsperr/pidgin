from pathlib import Path

import numpy as np
import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.evaluate import grid_rows, heuristic_choice, quick_metrics
from bridgezero.contract.model import ContractNet
from bridgezero.contract.targets import N_PAIR_ACTIONS, TorchScorer, pair_action_name
from bridgezero.contract.train import batch_losses, parse_args, run

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def smoke_deals():
    owners, tricks = load_dataset(SMOKE)
    return owners, tricks, TorchDeals(owners, tricks)


def test_pair_action_space_has_no_double_or_redouble():
    names = [pair_action_name(a) for a in range(N_PAIR_ACTIONS)]
    assert len(names) == 71 and names[-1] == "Pass"
    assert not any("X" in n for n in names)


def test_single_hand_inputs_ignore_unseen_cards_and_labels():
    owners, tricks, deals = smoke_deals()
    rng = np.random.default_rng(0)
    changed = owners.copy()
    for i in range(len(owners)):
        hidden = np.flatnonzero(owners[i] != 0)       # cards not held by North
        changed[i, hidden] = rng.permutation(owners[i, hidden])
    other = TorchDeals(changed, rng.integers(0, 14, tricks.shape))
    idx = torch.arange(len(owners))
    seat = torch.zeros(len(owners), dtype=torch.long)
    assert torch.equal(deals.inputs(idx, seat, "single"), other.inputs(idx, seat, "single"))
    assert not torch.equal(deals.inputs(idx, seat, "partnership"),
                           other.inputs(idx, seat, "partnership"))


def test_seat_rotation_preserves_inputs_labels_and_scores():
    owners, tricks, deals = smoke_deals()
    rotated = TorchDeals((owners + 1) % 4, np.roll(tricks, 1, axis=1))
    scorer = TorchScorer()
    idx = torch.arange(len(owners)).repeat(4)
    seat = torch.arange(4).repeat_interleave(len(owners))
    vul = (idx % 2)
    for mode in ("single", "partnership"):
        assert torch.equal(deals.inputs(idx, seat, mode), rotated.inputs(idx, (seat + 1) % 4, mode))
    rel = deals.rel_tricks(idx, seat)
    assert torch.equal(rel, rotated.rel_tricks(idx, (seat + 1) % 4))
    assert torch.equal(scorer.exact(rel, vul), scorer.exact(rotated.rel_tricks(idx, (seat + 1) % 4), vul))


def test_model_shapes_and_heuristics_run():
    _, _, deals = smoke_deals()
    rows = grid_rows(8)
    for mode in ("single", "partnership"):
        net = ContractNet(mode, width=32, suit_width=8, depth=1)
        out = net(deals.inputs(rows[0], rows[1], mode), rows[2])
        assert out["trick_logits"].shape == (64, 2, 5, 14)
        assert out["contract_q"].shape == (64, 71)
        choice = heuristic_choice(deals, rows, mode)
        assert choice.min() >= 0 and choice.max() <= 70


def test_tiny_partnership_overfit_reduces_loss_and_varies_choices():
    torch.manual_seed(0)
    _, _, deals = smoke_deals()
    scorer = TorchScorer()
    net = ContractNet("partnership", width=128, suit_width=32, depth=2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rows = grid_rows(deals.n)
    before = quick_metrics(net, deals, scorer)
    for _ in range(300):
        pick = torch.randint(len(rows[0]), (256,))
        losses = batch_losses(net, deals, rows[0][pick], rows[1][pick], rows[2][pick], scorer)
        opt.zero_grad()
        (losses["trick_nll"] + losses["q_loss"]).backward()
        opt.step()
    after = quick_metrics(net, deals, scorer)
    assert after["trick_nll"] < before["trick_nll"] - 1.0
    assert after["q_mae_points"] < before["q_mae_points"] * 0.6
    assert after["expected_score"] > 50  # beats passing everything on the fit set


def test_trainer_end_to_end_writes_metadata(tmp_path):
    args = parse_args([
        "--stage", "C", "--data", str(SMOKE), "--out", str(tmp_path),
        "--train-start", "0", "--train-count", "96", "--val-start", "96",
        "--val-count", "16", "--eval-start", "-16", "--eval-count", "16",
        "--steps", "4", "--batch", "32", "--eval-every", "2", "--width", "16",
        "--suit-width", "8", "--depth", "1", "--threads", "1"])
    report = run(args)
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ckpt["stage"] == "C" and ckpt["model_config"]["inputs"] == "single"
    assert report["rows"] == 16 * 8
    assert (tmp_path / "eval_rows.npz").exists()
    assert report["policies"]["dd_ceiling"]["mean_regret"] == 0.0
