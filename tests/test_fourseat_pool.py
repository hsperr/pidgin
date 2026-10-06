from pathlib import Path

import torch

from training.bridge.calls import DOUBLE
from training.bridge.deals import load_dataset
from training.contract.data import TorchDeals
from training.contract.model import AuctionContractNet
from training.contract.targets import TorchScorer
from training.fourseat.model import (
    FourSeatCritic,
    FourSeatDoubleGateNet,
    FourSeatNet,
    save_fourseat_checkpoint,
    warm_start_from_fourseat,
)
from training.fourseat.rollout import collect_trajectories, fourseat_validation

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
