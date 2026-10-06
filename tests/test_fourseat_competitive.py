"""Stage D5OWN4XC: Redouble and the Sacrifice gate, offered only where Pass ends the auction."""

from pathlib import Path

import numpy as np
import pytest
import torch

from training.bridge.auction import AuctionState
from training.bridge.calls import CONTRACTS, DOUBLE, PASS, REDOUBLE
from training.bridge.deals import load_dataset
from training.bridge.scoring import contract_score, stand_pat_actor_score, terminal_ns_score
from training.contract.data import TorchDeals
from training.contract.targets import TorchScorer
from training.fourseat.competitive import (
    FourSeatFinalDoubleBatch,
    FourSeatRedoubleBatch,
    collect_competitive_trajectories,
    competitive_trajectory_losses,
    competitive_validation,
    final_double_delta,
    redouble_delta,
    sac_targets,
)
from training.fourseat.model import (
    FourSeatCompetitiveCritic,
    FourSeatCompetitiveNet,
    FourSeatDoubleCritic,
    FourSeatDoubleGateNet,
    competitive_parts,
    competitive_path_log_probs,
    policy_log_probs,
    save_fourseat_checkpoint,
    warm_start_competitive,
)
from training.fourseat.state import FourSeatDoubleBatch, own_bid_scores, table_ns_score

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"
TRICK_STRAIN = (3, 2, 1, 0, 4)
ONE = torch.tensor([True])


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def final_legal_mask(ref: AuctionState) -> np.ndarray:
    """Reference legality with X/XX restricted to the pass-out seat."""
    expect = ref.legal_mask().copy()
    if not (ref.last_contract >= 0 and ref.pass_count == 2):
        expect[DOUBLE] = expect[REDOUBLE] = False
    return expect


def random_auctions(n, seed, silent_frac=0.0, bid_prob=0.3, double_prob=0.5, xx_prob=0.6):
    """Random legal auctions with X and XX, checked call by call against ``AuctionState``."""
    gen = torch.Generator().manual_seed(seed)
    deal = torch.randint(128, (n,), generator=gen)
    dealer = torch.randint(4, (n,), generator=gen)
    vns, vew = torch.randint(2, (n,), generator=gen), torch.randint(2, (n,), generator=gen)
    silent = torch.where(torch.rand(n, generator=gen) < silent_frac,
                         torch.randint(2, (n,), generator=gen), torch.full((n,), -1))
    batch = FourSeatRedoubleBatch.start(deal, dealer, vns, vew, silent)
    refs = [AuctionState(dealer=int(dealer[i]), vul_ns=bool(vns[i]), vul_ew=bool(vew[i]))
            for i in range(n)]
    snaps = []
    while not bool(batch.ended.all()):
        legal = batch.legal()
        assert legal.shape[1] == 38
        assert not bool((legal[:, DOUBLE] & legal[:, REDOUBLE]).any())
        for i in range(n):
            if refs[i].ended:
                assert not legal[i].any()
                continue
            expect = final_legal_mask(refs[i])
            if int(silent[i]) == refs[i].side_to_act:
                expect[:35] = False
                expect[DOUBLE] = expect[REDOUBLE] = False
            assert np.array_equal(legal[i].numpy(), expect), refs[i].format_history()
        snaps.append((batch.subset(torch.arange(n)), list(refs)))
        action = torch.full((n,), PASS)
        for i in range(n):
            u = float(torch.rand(1, generator=gen))
            options = legal[i, :35].nonzero().squeeze(1)
            if legal[i, REDOUBLE] and u < xx_prob:
                action[i] = REDOUBLE
            elif legal[i, DOUBLE] and u < double_prob:
                action[i] = DOUBLE
            elif len(options) and u < bid_prob:
                action[i] = options[torch.randint(min(len(options), 6), (1,), generator=gen)]
        batch.apply(action, ~batch.ended)
        for i in range(n):
            if not refs[i].ended:
                refs[i] = refs[i].apply(int(action[i]))
        assert [r.ended for r in refs] == batch.ended.tolist()
    return batch, refs, snaps


def start_one(calls, dealer=0, vul_ns=0, vul_ew=0, deal=0, cls=FourSeatRedoubleBatch):
    b = cls.start(torch.tensor([deal]), torch.tensor([dealer]), torch.tensor([vul_ns]),
                  torch.tensor([vul_ew]))
    for call in calls:
        b.apply(torch.tensor([call]), ONE)
    return b


def test_final_only_legality_features_and_termination():
    batch, refs, snaps = random_auctions(96, 21, silent_frac=0.2)
    assert batch.call_lists() == [list(r.calls) for r in refs]
    assert any(REDOUBLE in r.calls for r in refs) and any(DOUBLE in r.calls for r in refs)
    seen_final = seen_xx = False
    for state, state_refs in snaps:
        f = state.features()
        assert f.shape[1] == 151
        for i, ref in enumerate(state_refs):
            if ref.ended:
                continue
            final = ref.last_contract >= 0 and ref.pass_count == 2
            assert f[i, 149] == float(final)
            assert f[i, 150] == float(ref.doubled == 2)
            assert f[i, 147] == float(ref.doubled >= 1)
            seen_final |= final
            seen_xx |= ref.doubled == 2
    assert seen_final and seen_xx
    # direct seat cannot double; pass-out seat can; XX only at the declaring side's pass-out seat
    b = start_one([0])                                   # N 1C
    assert not b.legal()[0, DOUBLE]                      # E is in the direct seat
    b = start_one([0, PASS])
    assert not b.legal()[0, DOUBLE]                      # S is partner
    b = start_one([0, PASS, PASS])
    assert b.legal()[0, DOUBLE]                          # W: Pass would end the auction
    b.apply(torch.tensor([DOUBLE]), ONE)
    assert not b.legal()[0, REDOUBLE]                    # N is the direct seat after X
    b.apply(torch.tensor([PASS]), ONE)
    b.apply(torch.tensor([PASS]), ONE)
    assert b.legal()[0, REDOUBLE] and not b.legal()[0, DOUBLE]   # S may XX
    b.apply(torch.tensor([REDOUBLE]), ONE)
    for _ in range(2):
        b.apply(torch.tensor([PASS]), ONE)
    assert not b.ended[0] and not b.legal()[0, DOUBLE]
    b.apply(torch.tensor([1]), ONE)                      # E 1D clears X/XX
    assert not b.doubled[0] and not b.redoubled[0]
    for _ in range(3):
        b.apply(torch.tensor([PASS]), ONE)
    assert b.ended[0]
    # sacrifice-only batch: 37 actions, same final-only X
    s = start_one([0, PASS, PASS], cls=FourSeatFinalDoubleBatch)
    assert s.legal().shape[1] == 37 and s.legal()[0, DOUBLE]
    assert not start_one([0], cls=FourSeatFinalDoubleBatch).legal()[0, DOUBLE]


def test_redoubled_table_scores_and_deltas_match_reference():
    data = deals()
    tricks = data.tricks.numpy()
    batch, refs, snaps = random_auctions(160, 22)
    table = table_ns_score(batch, data)
    final = final_double_delta(batch, data)
    redoubled_rows = 0
    for i, ref in enumerate(refs):
        t = tricks[int(batch.deal[i])]
        assert table[i] == terminal_ns_score(ref, t)
        if ref.last_contract >= 0 and ref.doubled:
            _, level, strain = CONTRACTS[ref.last_contract]
            decl = ref.declarer()
            k = int(t[decl, TRICK_STRAIN[strain]])
            vul = ref.vulnerable(decl % 2)
            expect = (contract_score(level, strain, k, 0, vul)
                      - contract_score(level, strain, k, ref.doubled, vul))
            assert final[i] * 100 == pytest.approx(expect)
            redoubled_rows += ref.doubled == 2
        else:
            assert final[i] == 0
    assert redoubled_rows > 3
    checked = 0
    for state, state_refs in snaps:
        delta = redouble_delta(state, data)
        can = state.can_redouble()
        for i, ref in enumerate(state_refs):
            if bool(can[i]):
                t = tricks[int(state.deal[i])]
                expect = stand_pat_actor_score(ref, REDOUBLE, t) - stand_pat_actor_score(ref, PASS, t)
                assert delta[i] * 100 == pytest.approx(expect)
                checked += 1
    assert checked > 10


def _sac_reference(ref, cand, tricks):
    """Actor-side score of ``cand`` doubled by LHO then all pass, minus Pass now."""
    side = ref.side_to_act
    state = ref.apply(cand).apply(DOUBLE)
    while not state.ended:
        state = state.apply(PASS)
    ns = terminal_ns_score(state, tricks)
    return (ns if side == 0 else -ns) - stand_pat_actor_score(ref, PASS, tricks)


def test_sac_targets_match_reference_and_hand_built_deals():
    data = deals()
    tricks = data.tricks.numpy()
    _, _, snaps = random_auctions(128, 23, xx_prob=0.3)
    checked = 0
    for state, state_refs in snaps:
        cand, valid, delta, _ = sac_targets(state, data)
        for i, ref in enumerate(state_refs):
            spot = (not ref.ended and ref.last_contract >= 0 and ref.pass_count == 2
                    and ref.contract_seat % 2 != ref.side_to_act)
            if not spot:
                assert not valid[i].any()
                continue
            t = tricks[int(state.deal[i])]
            for k in range(5):
                expect = next((c for c in range(max(ref.last_contract + 1, 15), 35) if c % 5 == k), -1)
                assert int(cand[i, k]) == expect
                if expect < 0:
                    assert not valid[i, k]
                    continue
                assert bool(valid[i, k])
                assert delta[i, k] * 100 == pytest.approx(_sac_reference(ref, expect, t))
                checked += 1
    assert checked > 40
    t = tricks[5]
    # N 1S, E P, S 4S, W P, N P, E to act (pass-out seat): candidates 5C 5D 5H 5S 4NT
    b = start_one([3, PASS, 18, PASS, PASS], vul_ew=1, deal=5)
    cand, valid, delta, baseline = sac_targets(b, data)
    assert cand[0].tolist() == [20, 21, 22, 23, 19] and bool(valid.all())
    ns_4s = contract_score(4, 3, int(t[0, 0]), 0, False)          # N declares spades
    assert float(baseline[0]) == -ns_4s
    for k, c in enumerate([20, 21, 22, 23, 19]):
        level, strain = c // 5 + 1, c % 5
        ew = contract_score(level, strain, int(t[1, TRICK_STRAIN[strain]]), 1, True)  # E declares
        assert delta[0, k] * 100 == pytest.approx(ew + ns_4s)
    # low contract: N 1S P P, E to act -> level-4 candidates only
    cand, valid, _, _ = sac_targets(start_one([3, PASS, PASS], deal=5), data)
    assert cand[0].tolist() == [15, 16, 17, 18, 19]
    # not the pass-out seat: no spot
    _, valid, _, _ = sac_targets(start_one([3], deal=5), data)
    assert not valid.any()
    # nothing above 7NT except higher strains: over 7S only 7NT
    cand, _, _, _ = sac_targets(start_one([33, PASS, PASS], deal=5), data)
    assert cand[0].tolist() == [-1, -1, -1, -1, 34]


def _random_competitive(width=16, redouble=True, sacrifice=True, bound=None, seed=0):
    torch.manual_seed(seed)
    net = FourSeatCompetitiveNet(width, 8, 1, bound, redouble=redouble, sacrifice=sacrifice)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net


def test_gates_take_mass_only_from_pass_and_normalize():
    data = deals()
    net = _random_competitive()
    _, _, snaps = random_auctions(128, 24, xx_prob=0.3)
    counts = {"sac": 0, "x": 0, "xx": 0}
    for state, _ in snaps[1:12]:
        hand = data.hands[state.deal, state.actor_seat]
        out = net(hand, state.features())
        legal = state.legal()
        cand, valid, _, _ = sac_targets(state, data, legal)
        assert torch.equal(torch.where(valid, cand, -1), torch.where(valid, out["sac_candidates"], -1))
        for temp in (1.0, 0.7):
            logp = policy_log_probs(out, legal, temp)
            live = legal.any(1)
            assert torch.allclose(logp[live].exp().sum(1), torch.ones(int(live.sum())), atol=1e-5)
            part = competitive_parts(out, legal, temp)
            n = len(legal)
            for i in (part["can_d"] | part["spot"]).nonzero().squeeze(1).tolist():
                sm = part["rest"][i].exp()
                pd = torch.sigmoid(part["d_gate"][i]) if part["can_d"][i] else torch.tensor(0.0)
                ps = torch.sigmoid(out["sac_gate"][i] / temp) if part["spot"][i] else torch.tensor(0.0)
                expect = sm.clone()
                expect[PASS] = sm[PASS] * (1 - pd) * (1 - ps)
                if part["spot"][i]:
                    ch = int(part["chosen"][i])
                    best = out["sac_value"][i].masked_fill(~valid[i], -torch.inf)
                    assert ch == int(cand[i][best == best.max()].min())
                    expect[ch] += sm[PASS] * (1 - pd) * ps
                    counts["sac"] += 1
                assert torch.allclose(logp[i, :PASS + 1].exp(), expect, atol=1e-6)
                if part["can_x"][i]:
                    assert torch.allclose(logp[i, DOUBLE].exp(), sm[PASS] * pd, atol=1e-6)
                    counts["x"] += 1
                if part["can_xx"][i]:
                    assert torch.allclose(logp[i, REDOUBLE].exp(), sm[PASS] * pd, atol=1e-6)
                    counts["xx"] += 1
            # latent paths sum to the marginal: SAC candidate, Pass, X/XX
            spot = part["spot"]
            ch = part["chosen"].clamp(min=0)
            paths = {}
            for key, act, fired in (("soft", ch, False), ("fired", ch, True)):
                trunk, gate = competitive_path_log_probs(out, legal, act, torch.full((n,), fired), temp)
                paths[key] = (trunk + gate).exp()
            rows = torch.arange(n)
            assert torch.allclose((paths["soft"] + paths["fired"])[spot],
                                  logp[rows, ch].exp()[spot], atol=1e-6)
            for act in (PASS, DOUBLE, REDOUBLE):
                ok = legal[:, act]
                trunk, gate = competitive_path_log_probs(out, legal, torch.full((n,), act),
                                                         torch.zeros(n, dtype=torch.bool), temp)
                assert torch.allclose((trunk + gate).exp()[ok], logp[ok, act].exp(), atol=1e-6)
    assert counts["sac"] > 5 and counts["x"] > 5 and counts["xx"] > 0


def _gate_checkpoint(tmp_path, width=24, bound=None):
    torch.manual_seed(3)
    net, critic = FourSeatDoubleGateNet(width, 8, 2, bound), FourSeatDoubleCritic(width, 8, 2)
    with torch.no_grad():
        for p in (*net.parameters(), *critic.parameters()):
            p.normal_(0, 0.3)
        net.double_gate_head.bias.fill_(-1.0)
        net.policy_head.bias[DOUBLE] = 50.0         # unused column: must not change the bound
    path = tmp_path / "e20b" / "best.pt"
    save_fourseat_checkpoint(path, net, critic, step=11)
    return path, net, critic


@pytest.mark.parametrize("bound", [None, 2.0])
def test_warm_start_from_gate_checkpoint_is_identical_off_new_spots(tmp_path, bound):
    path, source, source_critic = _gate_checkpoint(tmp_path, bound=bound)
    actor, critic, meta = warm_start_competitive(path, xx_bias=-6.0, sac_bias=-6.0)
    assert meta["init_step"] == 11 and actor.config.get("policy_logit_bound") == bound
    data = deals()
    _, _, snaps = random_auctions(64, 25, xx_prob=0.4)
    compared = 0
    for state, _ in snaps[1:12]:
        hand = data.hands[state.deal, state.actor_seat]
        feats = state.features()
        a, b = source(hand, feats[:, :149]), actor(hand, feats)
        for key in ("trick_logits", "double_value", "double_gate"):
            assert torch.equal(a[key], b[key]), key
        assert torch.equal(a["policy_logits"], b["policy_logits"][:, :37])
        assert torch.equal(a["contract_q"], b["contract_q"][:, :37])
        legal = state.legal()
        parent_legal = torch.cat((legal[:, :DOUBLE], FourSeatDoubleBatch.can_double(state)[:, None]), 1)
        pa, pb = policy_log_probs(a, parent_legal), policy_log_probs(b, legal)
        part = competitive_parts(b, legal)
        plain = legal.any(1) & ~part["spot"] & ~part["can_d"] & ~parent_legal[:, DOUBLE]
        assert torch.equal(pa[plain], pb[plain, :37])
        compared += int(plain.sum())
        live = legal.any(1)
        assert torch.allclose(pb[live].exp().sum(1), torch.ones(int(live.sum())), atol=1e-5)
        pair = torch.stack((hand, data.hands[state.deal, (state.actor_seat + 2) % 4]), 1)
        assert torch.equal(source_critic(pair, feats[:, :149]), critic(pair, feats))
    assert compared > 50
    val = competitive_validation(actor, data, TorchScorer())
    assert val["redoubles"] == 0 and val["sacs"] == 0
    assert set(val["spots_per_1000"]) == {"x", "xx", "sac"}


def test_sacrifice_credit_own_bid_exclusion_and_detached_gates():
    data = deals()
    actor = _random_competitive(width=16, seed=4)
    critic = FourSeatCompetitiveCritic(16, 8, 1)
    with torch.no_grad():
        actor.policy_head.bias[PASS] = 2.0                    # reach pass-out seats
        for head, bias in ((actor.sac_gate_head, 2.0), (actor.redouble_gate_head, 2.0),
                           (actor.double_gate_head, 0.0)):
            head.weight.zero_()
            head.bias.fill_(bias)
    traj = collect_competitive_trajectories(actor, data, 256, torch.Generator().manual_seed(5),
                                            TorchScorer(), silent_frac=0.0)
    fired = traj.sac_fired
    assert bool(fired.any()) and bool((traj.standing_xx >= 0).any())
    tricks = data.tricks.numpy()
    term = traj.terminal
    calls = term.call_lists()
    refs = [AuctionState.from_calls(c, int(term.dealer[i]), bool(term.vul[i, 0]), bool(term.vul[i, 1]))
            for i, c in enumerate(calls)]
    for j in fired.nonzero().squeeze(1).tolist():
        i, t = int(traj.row[j]), int(traj.states.t[j])
        before = AuctionState.from_calls(calls[i][:t], int(term.dealer[i]),
                                         bool(term.vul[i, 0]), bool(term.vul[i, 1]))
        assert before.pass_count == 2 and int(traj.actions[j]) >= 15
        side = before.side_to_act
        final_ns = terminal_ns_score(refs[i], tricks[int(term.deal[i])])
        ns_then = stand_pat_actor_score(before, PASS, tricks[int(term.deal[i])])
        assert traj.sac_credit[j] * 100 == pytest.approx((final_ns if side == 0 else -final_ns) - ns_then)
    for i, ref in enumerate(refs):                             # every X/XX was at a pass-out seat
        state = AuctionState(dealer=ref.dealer, vul_ns=ref.vul_ns, vul_ew=ref.vul_ew)
        for call in ref.calls:
            if call in (DOUBLE, REDOUBLE):
                assert state.pass_count == 2
            state = state.apply(call)
    exclude = torch.zeros(len(term), 35, dtype=torch.bool)
    exclude[traj.row[fired], traj.actions[fired]] = True
    score, _, _ = own_bid_scores(term, data, TorchScorer(), exclude=exclude)
    assert torch.equal(score, traj.side_score)
    plain, _, _ = own_bid_scores(term, data, TorchScorer())
    assert not torch.equal(score, plain)
    losses = competitive_trajectory_losses(actor, critic, data, traj, all_spots=True)
    keys = ("sac_value_loss", "sac_ce", "redouble_value_loss", "redouble_ce",
            "double_value_loss", "double_ce")
    for key in keys:
        assert float(losses[key].detach()) > 0, key
    trunk = [*actor.suit_net.parameters(), *actor.auction_net.parameters(), *actor.trunk.parameters()]
    for key in keys:
        actor.zero_grad(set_to_none=True)
        losses[key].backward(retain_graph=True)
        assert all(p.grad is None or not p.grad.any() for p in trunk), key
    gates = (actor.sac_gate_head, actor.redouble_gate_head, actor.double_gate_head)
    actor.zero_grad(set_to_none=True)
    losses["policy_objective"].backward(retain_graph=True)
    for head in gates:                                        # default: PG never trains gates
        assert head.bias.grad is None or float(head.bias.grad.abs()) == 0.0
    gated = competitive_trajectory_losses(actor, critic, data, traj, gate_pg=True)
    actor.zero_grad(set_to_none=True)
    gated["policy_objective"].backward()
    assert all(float(head.bias.grad.abs()) > 0 for head in gates)
    assert all(bool(p.grad.isfinite().all()) for p in actor.parameters() if p.grad is not None)
    # a fired SAC's gate log-prob never reaches the trunk
    out = actor(data.hands[traj.states.deal, traj.states.actor_seat], traj.states.features())
    _, gate_lp = competitive_path_log_probs(out, traj.states.legal(), traj.actions, fired,
                                            detach_gates=False)
    actor.zero_grad(set_to_none=True)
    gate_lp[fired].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in trunk)
    assert 0 < float(losses["clean_share"]) <= 1
    default = competitive_trajectory_losses(actor, critic, data, traj)
    assert default["sac_clean_decisions"] <= default["sac_decisions"]


def test_sacrifice_only_net_uses_final_double_batch(tmp_path):
    path, _, _ = _gate_checkpoint(tmp_path, width=16)
    actor, critic, _ = warm_start_competitive(path, redouble=False, sacrifice=True, sac_bias=-1.0)
    assert actor.n_actions == 37 and actor.extra_features == 74
    data = deals()
    traj = collect_competitive_trajectories(actor, data, 32, torch.Generator().manual_seed(6),
                                            TorchScorer(), silent_frac=0.0)
    assert type(traj.terminal) is FourSeatFinalDoubleBatch
    losses = competitive_trajectory_losses(actor, critic, data, traj)
    assert "sac_value_loss" in losses and "redouble_value_loss" not in losses
    val = competitive_validation(actor, data, TorchScorer())
    assert "sac_rate" in val and val["spots_per_1000"]["xx"] == 0


def test_table_weight_mixes_real_table_result_into_returns():
    data = deals()
    actor = _random_competitive(width=16, seed=4)
    critic = FourSeatCompetitiveCritic(16, 8, 1)
    with torch.no_grad():
        actor.policy_head.bias[PASS] = 2.0
        for head, bias in ((actor.sac_gate_head, 2.0), (actor.redouble_gate_head, 2.0),
                           (actor.double_gate_head, 0.0)):
            head.weight.zero_()
            head.bias.fill_(bias)
    traj = collect_competitive_trajectories(actor, data, 256, torch.Generator().manual_seed(5),
                                            TorchScorer(), silent_frac=0.0)
    term = traj.terminal
    tricks = data.tricks.numpy()
    ns = torch.tensor([terminal_ns_score(AuctionState.from_calls(c, int(term.dealer[i]), bool(term.vul[i, 0]),
                                                                 bool(term.vul[i, 1])), tricks[int(term.deal[i])])
                       for i, c in enumerate(term.call_lists())], dtype=torch.float)
    side = torch.tensor([0, 1]).repeat(len(term))
    rows = torch.arange(len(term)).repeat_interleave(2)
    groups = torch.unique(rows * 2 + side)
    expect = torch.where(groups % 2 == 0, ns[groups // 2], -ns[groups // 2])
    assert torch.allclose(traj.table_score, expect)
    assert bool((traj.table_score != traj.score).any())         # opponents' contracts and doubles count
    plain = competitive_trajectory_losses(actor, critic, data, traj, all_spots=True)
    zero = competitive_trajectory_losses(actor, critic, data, traj, all_spots=True, table_weight=0.0)
    for key in ("policy_loss", "critic_loss", "q_loss"):
        assert torch.equal(plain[key], zero[key]), key
    full = competitive_trajectory_losses(actor, critic, data, traj, all_spots=True, table_weight=1.0)
    assert torch.allclose(full["return_mean"], ((traj.table_score - traj.ceiling) / 100).mean())
    assert full["sac_credit_mean"] == 0 and full["double_credit_mean"] == 0
    half = competitive_trajectory_losses(actor, critic, data, traj, all_spots=True, table_weight=0.5)
    assert not torch.equal(plain["policy_loss"], half["policy_loss"])
    with pytest.raises(ValueError):
        competitive_trajectory_losses(actor, critic, data, traj, table_weight=1.5)
