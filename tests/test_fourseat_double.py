from pathlib import Path

import numpy as np
import pytest
import torch

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import DOUBLE, PASS, REDOUBLE
from bridgezero.bridge.deals import load_dataset
from bridgezero.bridge.scoring import own_contract_score, stand_pat_actor_score, terminal_ns_score
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import AuctionContractNet, save_checkpoint
from bridgezero.contract.targets import TorchScorer
from bridgezero.cooperative.actor_critic import CentralCritic
from bridgezero.fourseat.double import double_trajectory_losses
from bridgezero.fourseat.model import (
    FourSeatDoubleNet,
    SilentView,
    load_fourseat_checkpoint,
    warm_start,
)
from bridgezero.fourseat.rollout import collect_trajectories, fourseat_validation
from bridgezero.fourseat.state import (
    FOURSEAT_DOUBLE_FEATURES,
    FOURSEAT_FEATURES,
    FourSeatDoubleBatch,
    double_delta,
    own_bid_scores,
    table_ns_score,
)
from bridgezero.fourseat.train import parse_args, run

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def slow_features(state: AuctionState, actor: int) -> np.ndarray:
    f = np.zeros(FOURSEAT_DOUBLE_FEATURES, dtype=np.float32)
    seen = False
    doubler = None
    for i, call in enumerate(state.calls):
        seat = (state.dealer + i) % 4
        rel = (seat - actor) % 4
        if call < PASS:
            f[{0: 0, 2: 35, 1: 77, 3: 112}[rel] + call] = 1
            seen |= rel % 2 == 0
            doubler = None
        elif call == DOUBLE:
            doubler = rel
        elif rel % 2 == 0 and not seen:
            f[70 if rel == 0 else 71] = 1
    f[72 + (state.dealer - actor) % 4] = 1
    f[76] = float(state.vulnerable(actor % 2))
    if doubler is not None:
        f[147] = 1
        f[148] = float(doubler % 2 == 0)
    return f


def random_auctions(n, seed, silent_frac=0.0, bid_prob=0.4, double_prob=0.35):
    """Random legal auctions with Double; checks legality/termination vs AuctionState."""
    gen = torch.Generator().manual_seed(seed)
    deal = torch.randint(128, (n,), generator=gen)
    dealer = torch.randint(4, (n,), generator=gen)
    vns, vew = torch.randint(2, (n,), generator=gen), torch.randint(2, (n,), generator=gen)
    silent = torch.where(torch.rand(n, generator=gen) < silent_frac,
                         torch.randint(2, (n,), generator=gen), torch.full((n,), -1))
    batch = FourSeatDoubleBatch.start(deal, dealer, vns, vew, silent)
    refs = [AuctionState(dealer=int(dealer[i]), vul_ns=bool(vns[i]), vul_ew=bool(vew[i]))
            for i in range(n)]
    snaps = []
    while not bool(batch.ended.all()):
        legal = batch.legal()
        assert legal.shape[1] == 37
        for i in range(n):
            if refs[i].ended:
                assert not legal[i].any()
                continue
            expect = refs[i].legal_mask()[:37].copy()     # REDOUBLE (37) never legal
            if int(silent[i]) == refs[i].side_to_act:
                expect[:35] = False
                expect[DOUBLE] = False
            assert np.array_equal(legal[i].numpy(), expect), refs[i].format_history()
        snaps.append((batch.subset(torch.arange(n)), list(refs)))
        action = torch.full((n,), PASS)
        for i in range(n):
            u = float(torch.rand(1, generator=gen))
            options = legal[i, :35].nonzero().squeeze(1)
            if legal[i, DOUBLE] and u < double_prob:
                action[i] = DOUBLE
            elif len(options) and u < double_prob + bid_prob:
                action[i] = options[torch.randint(min(len(options), 6), (1,), generator=gen)]
        batch.apply(action, ~batch.ended)
        for i in range(n):
            if not refs[i].ended:
                refs[i] = refs[i].apply(int(action[i]))
        assert [r.ended for r in refs] == batch.ended.tolist()
    return batch, refs, snaps


def test_double_legality_termination_and_no_redouble():
    batch, refs, _ = random_auctions(64, 11, silent_frac=0.25)
    assert batch.call_lists() == [list(r.calls) for r in refs]
    assert any(DOUBLE in r.calls for r in refs)
    # a Double followed by three passes ends the auction; X is not a pass
    b = FourSeatDoubleBatch.start(*(torch.tensor([v]) for v in (0, 0, 0, 0)))
    one = torch.tensor([True])
    b.apply(torch.tensor([0]), one)                   # N 1C
    assert b.legal()[0, DOUBLE]                       # E may double
    b.apply(torch.tensor([DOUBLE]), one)
    assert not b.legal()[0, DOUBLE]                   # S cannot redouble or double
    with pytest.raises(ValueError):
        b.apply(torch.tensor([REDOUBLE]), one)
    for _ in range(2):
        b.apply(torch.tensor([PASS]), one)
    assert not b.ended[0]
    b.apply(torch.tensor([PASS]), one)
    assert b.ended[0] and b.doubled[0]


def test_scores_and_delta_match_reference():
    data = deals()
    tricks = data.tricks.numpy()
    batch, refs, snaps = random_auctions(96, 12)
    score, _, _ = own_bid_scores(batch, data, TorchScorer())
    table = table_ns_score(batch, data)
    for i, ref in enumerate(refs):
        t = tricks[int(batch.deal[i])]
        assert table[i] == terminal_ns_score(ref, t)
        for side in (0, 1):
            assert score[i, side] == own_contract_score(ref, t, side)
    checked = 0
    for state, state_refs in snaps:
        delta = double_delta(state, data)
        can = state.can_double()
        for i, ref in enumerate(state_refs):
            if not bool(can[i]):
                continue
            t = tricks[int(state.deal[i])]
            expect = (stand_pat_actor_score(ref, DOUBLE, t) - stand_pat_actor_score(ref, PASS, t))
            assert delta[i] * 100 == pytest.approx(expect)
            checked += 1
    assert checked > 20


def test_double_features_match_slow_reference():
    _, _, snaps = random_auctions(48, 13, silent_frac=0.2)
    seen_doubled = False
    for batch, refs in snaps:
        feats = batch.features()
        assert feats.shape[1] == FOURSEAT_DOUBLE_FEATURES
        for i, ref in enumerate(refs):
            if ref.ended:
                continue
            assert np.array_equal(feats[i].numpy(), slow_features(ref, ref.turn))
            seen_doubled |= bool(feats[i, 147])
    assert seen_doubled


def _init_checkpoint(tmp_path, width=32):
    torch.manual_seed(0)
    base, critic = AuctionContractNet(width, 8, 2), CentralCritic(width, 8, 2)
    with torch.no_grad():
        for p in (*base.parameters(), *critic.parameters()):
            p.normal_(0, 0.3)
    path = tmp_path / "init" / "best.pt"
    save_checkpoint(path, base, "D4PG", step=1, critic_config=critic.config,
                    critic=critic.state_dict())
    return path, base


def test_double_warm_start_matches_no_double_model(tmp_path):
    path, base = _init_checkpoint(tmp_path)
    plain, _, _ = warm_start(path)
    dbl, critic, _ = warm_start(path, doubles=True, double_bias=-3.0)
    assert float(dbl.policy_head.bias.detach()[DOUBLE]) == -3.0
    batch, _, snaps = random_auctions(64, 14, double_prob=0.0)   # opponents bid, nobody doubles
    data = deals()
    for state, _ in snaps[:6]:
        hand = data.hands[state.deal, state.actor_seat]
        f = state.features()
        a, b = plain(hand, f[:, :FOURSEAT_FEATURES]), dbl(hand, f)
        assert torch.equal(a["policy_logits"], b["policy_logits"][:, :36])
        assert torch.equal(a["contract_q"], b["contract_q"][:, :36])
        assert torch.equal(a["trick_logits"], b["trick_logits"])
        assert not b["contract_q"][:, DOUBLE].any()
    # silent view == the cooperative source on 77 features
    hand = data.hands[:8, 0]
    auction = torch.zeros(8, 77)
    auction[:, 72] = 1
    assert torch.equal(SilentView(dbl)(hand, auction)["policy_logits"],
                       base(hand, auction)["policy_logits"])


def test_standing_double_credit_and_counterfactual_gradient():
    data = deals()
    actor = FourSeatDoubleNet(16, 8, 1)
    from bridgezero.fourseat.model import FourSeatDoubleCritic
    critic = FourSeatDoubleCritic(16, 8, 1)
    with torch.no_grad():
        actor.policy_head.bias[DOUBLE] = 1.0     # double often so credit is exercised
    traj = collect_trajectories(actor, data, 64, torch.Generator().manual_seed(15), TorchScorer(),
                                silent_frac=0.0, doubles=True)
    assert bool((traj.actions == DOUBLE).any()) and bool((traj.standing_x >= 0).any())
    losses = double_trajectory_losses(actor, critic, data, traj)
    assert losses["double_cf_decisions"] > 0
    stand = (traj.actions == DOUBLE) & (traj.states.t == traj.standing_x[traj.row])
    # credited doubles are on the opponents' contract and are the last X of their episode
    assert bool(stand.any())
    assert bool((traj.states.side[stand] != traj.terminal.contract_seat[traj.row[stand]] % 2).all())
    actor.zero_grad()
    losses["double_cf_loss"].backward()
    assert float(actor.policy_head.bias.grad[DOUBLE]) != 0.0
    val = fourseat_validation(actor, data, TorchScorer(), doubles=True)
    assert 0.0 <= val["double_rate"] <= 1.0 and "p_double_final" in val


def test_double_trainer_smoke(tmp_path):
    init, _ = _init_checkpoint(tmp_path, width=16)
    out = tmp_path / "e19"
    report = run(parse_args([
        "--data", str(SMOKE), "--out", str(out), "--init", str(init), "--doubles",
        "--train-start", "0", "--train-count", "96", "--val-start", "96", "--val-count", "16",
        "--eval-start", "112", "--eval-count", "16", "--steps", "2", "--episodes", "16",
        "--eval-every", "1", "--threads", "1", "--max-double-rate", "1.1"]))
    assert "double_rate" in report["fourseat"]
    net, meta = load_fourseat_checkpoint(out / "best.pt")
    assert meta["stage"] == "D5OWN4X" and isinstance(net, FourSeatDoubleNet)


def test_double_value_warm_start_and_nonvanishing_policy_gradient(tmp_path):
    from bridgezero.fourseat.model import FourSeatDoubleCritic, FourSeatDoubleValueNet
    path, _ = _init_checkpoint(tmp_path)
    plain, _, _ = warm_start(path, doubles=True)
    value_net, _, _ = warm_start(path, double_value=True)
    assert isinstance(value_net, FourSeatDoubleValueNet)
    data = deals()
    _, _, snaps = random_auctions(32, 16)
    state = snaps[4][0]
    hand = data.hands[state.deal, state.actor_seat]
    a, b = plain(hand, state.features()), value_net(hand, state.features())
    assert torch.equal(a["policy_logits"], b["policy_logits"])
    assert not b["double_value"].any()
    # small p(X) and a positive learned value: the CE still pushes the X logit up
    actor, critic = FourSeatDoubleValueNet(16, 8, 1), FourSeatDoubleCritic(16, 8, 1)
    with torch.no_grad():
        actor.policy_head.bias[DOUBLE] = -8.0
        actor.policy_head.bias[:35] = 2.0
        actor.double_value_head.bias.fill_(3.0)
    traj = collect_trajectories(actor, data, 64, torch.Generator().manual_seed(17), TorchScorer(),
                                silent_frac=0.0, doubles=True)
    losses = double_trajectory_losses(actor, critic, data, traj, tau=0.5)
    assert losses["double_decisions_all"] >= losses.get("double_decisions_final", 0)
    actor.zero_grad()
    losses["double_ce"].backward()
    assert float(actor.policy_head.bias.grad[DOUBLE]) < -1e-3
    val = fourseat_validation(actor, data, TorchScorer(), doubles=True)
    assert len(val["double_value"]["deciles"]) > 0


def test_double_value_trainer_smoke(tmp_path):
    from bridgezero.fourseat.model import FourSeatDoubleValueNet
    init, _ = _init_checkpoint(tmp_path, width=16)
    out = tmp_path / "e19b"
    report = run(parse_args([
        "--data", str(SMOKE), "--out", str(out), "--init", str(init), "--double-value",
        "--train-start", "0", "--train-count", "96", "--val-start", "96", "--val-count", "16",
        "--eval-start", "112", "--eval-count", "16", "--steps", "3", "--episodes", "16",
        "--double-policy-start", "1", "--eval-every", "1", "--threads", "1",
        "--max-double-rate", "1.1"]))
    assert "double_rate" in report["fourseat"]
    net, meta = load_fourseat_checkpoint(out / "best.pt")
    assert meta["stage"] == "D5OWN4XV" and isinstance(net, FourSeatDoubleValueNet)


def test_double_gate_never_writes_to_trunk(tmp_path):
    from bridgezero.fourseat.model import FourSeatDoubleCritic, FourSeatDoubleGateNet, policy_log_probs
    path, base = _init_checkpoint(tmp_path)
    plain, _, _ = warm_start(path, doubles=True)
    gate_net, _, _ = warm_start(path, double_gate=True, double_bias=-6.0)
    data = deals()
    _, _, snaps = random_auctions(48, 18)
    state = snaps[5][0]
    hand = data.hands[state.deal, state.actor_seat]
    legal = state.legal()
    a, b = plain(hand, state.features()), gate_net(hand, state.features())
    assert torch.equal(a["policy_logits"][:, :36], b["policy_logits"][:, :36])
    logp = policy_log_probs(b, legal)
    live = legal.any(1)
    assert torch.allclose(logp[live].exp().sum(1), torch.ones(int(live.sum())), atol=1e-5)
    can = legal[:, DOUBLE]
    assert bool(can.any())
    assert torch.allclose(logp[can, DOUBLE].exp(), torch.sigmoid(torch.tensor(-6.0)).expand(int(can.sum())))
    # losses from a doubling-heavy batch: X-only terms leave the trunk untouched
    actor, critic = FourSeatDoubleGateNet(16, 8, 1), FourSeatDoubleCritic(16, 8, 1)
    with torch.no_grad():
        actor.double_gate_head.bias.fill_(0.0)
        actor.double_value_head.bias.fill_(1.0)
    traj = collect_trajectories(actor, data, 64, torch.Generator().manual_seed(19), TorchScorer(),
                                silent_frac=0.0, doubles=True)
    assert bool((traj.actions == DOUBLE).any())
    losses = double_trajectory_losses(actor, critic, data, traj)
    trunk = [*actor.suit_net.parameters(), *actor.auction_net.parameters(), *actor.trunk.parameters()]
    x_only = traj.actions == DOUBLE
    out = actor(data.hands[traj.states.deal, traj.states.actor_seat], traj.states.features())
    logp_x = policy_log_probs(out, traj.states.legal())[x_only, DOUBLE].sum()
    for loss in (losses["double_ce"], losses["double_value_loss"], logp_x):
        actor.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        assert all(p.grad is None or not p.grad.any() for p in trunk)
    assert actor.double_gate_head.bias.grad is not None
    val = fourseat_validation(actor, data, TorchScorer(), doubles=True)
    assert "double_value" in val


def test_double_value_margin_rule():
    from bridgezero.fourseat.model import FourSeatDoubleGateNet
    data = deals()
    actor = FourSeatDoubleGateNet(16, 8, 1)
    never = fourseat_validation(actor, data, TorchScorer(), doubles=True, double_margin=100.0)
    always = fourseat_validation(actor, data, TorchScorer(), doubles=True, double_margin=-100.0)
    assert never["doubles"] == 0
    assert always["double_rate"] > 0.5
