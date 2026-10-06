"""Fast-rollout equivalence with the default four-seat rollout (CPU).

The fast path samples by inverse CDF from pre-drawn uniforms instead of
``torch.multinomial``. Driving the old ``play`` loop with the same uniforms must
reproduce the fast trajectories exactly: same states (incl. history, legal masks,
features), same actions, groups, scores, and the same losses.
"""

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from training.bridge.calls import DOUBLE, PASS, REDOUBLE
from training.bridge.deals import load_dataset
from training.contract.data import TorchDeals
from training.contract.model import AuctionContractNet
from training.contract.targets import TorchScorer
from training.fourseat.competitive import (
    MAX_REDOUBLE_CALLS,
    CompetitiveTrajectories,
    competitive_batch_class,
    competitive_trajectories,
    competitive_trajectory_losses,
)
from training.fourseat.fast_rollout import (
    COMPETITIVE_ROUNDS,
    ROUNDS,
    FastCollector,
    inverse_cdf_sample,
    sac_fire,
)
from training.fourseat.model import (
    FourSeatCompetitiveCritic,
    FourSeatCompetitiveNet,
    FourSeatDoubleGateNet,
    FourSeatNet,
    competitive_parts,
    policy_log_probs,
)
from training.fourseat.rollout import (
    FourSeatTrajectories,
    batch_class,
    episode_setup,
    play,
)
from training.fourseat.state import double_delta, own_bid_scores

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def noisy(net, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    return net


def teacher_forced_old(actor, data, episodes, generator, scorer, silent_frac, doubles, pool,
                       temperature=1.0):
    """``rollout.collect_trajectories`` with multinomial replaced by the fast path's uniforms."""
    deal, dealer, vns, vew, opponent, frozen_side, silent = episode_setup(
        data, episodes, generator, silent_frac, "cpu", pool)
    u = torch.rand((ROUNDS, episodes), generator=generator)
    batch = batch_class(doubles).start(deal, dealer, vns, vew, silent)
    frozen = None
    if pool:
        frozen = {"players": {c: n for c, (n, _) in pool.items()}, "code": opponent,
                  "side": frozen_side}

    def choose(log_probs):
        decide = ~batch.ended & ~batch.forced
        if frozen is not None:
            decide &= frozen["side"] != batch.side
        ids = decide.nonzero().squeeze(1)
        return inverse_cdf_sample(log_probs, u[batch.t[ids], ids], log_probs.isfinite())

    record = []
    play(actor, data, batch, choose, chunk=1 << 30, temperature=temperature, frozen=frozen,
         on_decision=lambda idx, state, chosen, *_: record.append((idx, state, chosen)))
    states = type(batch).cat([s for _, s, _ in record])
    actions = torch.cat([c for _, _, c in record])
    rows = torch.cat([i for i, _, _ in record])
    present, dense = torch.unique(rows * 2 + states.side, return_inverse=True)
    score, ceiling, _ = own_bid_scores(batch, data, scorer)
    standing_x = batch.standing_double_position() if doubles else torch.full_like(deal, -1)
    final_delta = double_delta(batch, data) if doubles else torch.zeros(episodes)
    return FourSeatTrajectories(states, actions, dense, score.reshape(-1)[present],
                                ceiling.reshape(-1)[present], batch, score, ceiling, rows,
                                standing_x, final_delta, opponent, frozen_side)


def assert_same(fast, old):
    for f in fields(old.states):
        assert torch.equal(getattr(fast.states, f.name), getattr(old.states, f.name)), f.name
        assert torch.equal(getattr(fast.terminal, f.name), getattr(old.terminal, f.name)), f.name
    for name in ("actions", "episode", "score", "ceiling", "side_score", "side_ceiling", "row",
                 "standing_x", "final_delta", "opponent", "frozen_side"):
        assert torch.equal(getattr(fast, name), getattr(old, name)), name
    assert torch.equal(fast.states.legal(), old.states.legal())
    assert torch.equal(fast.states.features(), old.states.features())


def teacher_forced_old_competitive(actor, data, episodes, generator, scorer, silent_frac, pool,
                                   temperature=1.0):
    """``collect_competitive_trajectories`` with multinomial and the SAC fire draw replaced
    by the fast path's uniforms (independent re-implementation of the default loop)."""
    deal, dealer, vns, vew, opponent, frozen_side, silent = episode_setup(
        data, episodes, generator, silent_frac, "cpu", pool)
    u = torch.rand((COMPETITIVE_ROUNDS, episodes), generator=generator)
    u_fire = torch.rand((COMPETITIVE_ROUNDS, episodes), generator=generator) if actor.sacrifice else None
    batch = competitive_batch_class(actor.redouble).start(deal, dealer, vns, vew, silent)
    frozen = None
    if pool:
        frozen = {"players": {c: n for c, (n, _) in pool.items()}, "code": opponent,
                  "side": frozen_side}
    top = torch.ones(episodes, MAX_REDOUBLE_CALLS, dtype=torch.bool)

    def choose(log_probs):
        decide = ~batch.ended & ~batch.forced
        if frozen is not None:
            decide &= frozen["side"] != batch.side
        ids = decide.nonzero().squeeze(1)
        return inverse_cdf_sample(log_probs, u[batch.t[ids], ids], log_probs.isfinite())

    record = []

    def on_decision(idx, state, chosen, masked, outputs):
        top[idx, state.t] = chosen == masked.argmax(-1)
        fired = torch.zeros_like(chosen, dtype=torch.bool)
        if "sac_gate" in outputs:
            part = competitive_parts(outputs, state.legal(), temperature)
            rows = torch.arange(len(chosen))
            soft = part["rest"][rows, chosen.clamp(max=PASS)]
            fire = part["sac_lp"] - torch.logaddexp(soft, part["sac_lp"])
            fired = part["spot"] & (chosen == part["chosen"]) & (u_fire[state.t, idx] < fire.exp())
            assert torch.equal(fired, sac_fire(outputs, state.legal(), chosen,
                                               u_fire[state.t, idx], temperature))
        record.append((idx, state, chosen, fired))

    play(actor, data, batch, choose, chunk=1 << 30, temperature=temperature, frozen=frozen,
         on_decision=on_decision)
    states = type(batch).cat([s for _, s, _, _ in record])
    actions = torch.cat([c for _, _, c, _ in record])
    rows = torch.cat([i for i, _, _, _ in record])
    fired = torch.cat([f for _, _, _, f in record])
    return competitive_trajectories(data, scorer, batch, states, actions, rows, fired, top,
                                    opponent, frozen_side)


CASES = {
    "no_double_bound": dict(doubles=False, bound=3.0, pool=False, silent=0.25),
    "double_gate_bound": dict(doubles=True, bound=3.0, pool=False, silent=0.25),
    "double_gate_nobound_pool": dict(doubles=True, bound=None, pool=True, silent=0.5),
}


@pytest.mark.parametrize("case", CASES)
def test_fast_rollout_matches_teacher_forced_old_path(case):
    cfg = CASES[case]
    data, scorer = deals(), TorchScorer()
    if cfg["doubles"]:
        actor = noisy(FourSeatDoubleGateNet(24, 8, 2, policy_logit_bound=cfg["bound"]), 1)
        with torch.no_grad():
            actor.double_gate_head.bias.fill_(0.5)      # doubles actually happen
    else:
        actor = noisy(FourSeatNet(24, 8, 2, policy_logit_bound=cfg["bound"]), 1)
    pool = ({1: (noisy(FourSeatNet(24, 8, 2), 3), 0.3), 2: (noisy(AuctionContractNet(24, 8, 2), 4), 0.3)}
            if cfg["pool"] else None)
    episodes = 96
    collector = FastCollector(actor, episodes, "cpu", cfg["doubles"], pool=pool, check_every=5)
    for seed in (11, 12):
        fast = collector.collect(data, torch.Generator().manual_seed(seed), scorer, cfg["silent"])
        old = teacher_forced_old(actor, data, episodes, torch.Generator().manual_seed(seed),
                                 scorer, cfg["silent"], cfg["doubles"], pool)
        assert_same(fast, old)
        assert len(fast) > episodes
    if cfg["doubles"]:
        assert int((fast.actions == DOUBLE).sum()) > 0


XC_CASES = {
    "xx_sac_bound": dict(redouble=True, sacrifice=True, bound=3.0, pool=False, silent=0.25),
    "xx_sac_bound_pool": dict(redouble=True, sacrifice=True, bound=3.0, pool=True, silent=0.5),
    "sac_only_nobound_pool": dict(redouble=False, sacrifice=True, bound=None, pool=True, silent=0.0),
    "xx_only_bound": dict(redouble=True, sacrifice=False, bound=3.0, pool=False, silent=0.0),
}


def competitive_actor(cfg):
    actor = noisy(FourSeatCompetitiveNet(24, 8, 2, policy_logit_bound=cfg["bound"],
                                         redouble=cfg["redouble"], sacrifice=cfg["sacrifice"]), 1)
    with torch.no_grad():                                 # reach pass-out seats; X/XX/SAC happen
        actor.policy_head.bias[PASS] += 2.0
        gates = [(actor.double_gate_head, 0.0)]
        if cfg["redouble"]:
            gates.append((actor.redouble_gate_head, 1.0))
        if cfg["sacrifice"]:
            gates.append((actor.sac_gate_head, 1.0))
        for head, bias in gates:
            head.weight.mul_(0.2)
            head.bias.fill_(bias)
    return actor


@pytest.mark.parametrize("case", XC_CASES)
def test_fast_competitive_rollout_matches_teacher_forced_old_path(case):
    cfg = XC_CASES[case]
    data, scorer = deals(), TorchScorer()
    actor = competitive_actor(cfg)
    critic = noisy(FourSeatCompetitiveCritic(24, 8, 2), 2)
    pool = ({1: (noisy(FourSeatNet(24, 8, 2), 3), 0.3),
             2: (noisy(AuctionContractNet(24, 8, 2), 4), 0.2),
             3: (noisy(FourSeatDoubleGateNet(24, 8, 2), 5), 0.1)}
            if cfg["pool"] else None)
    episodes = 128
    collector = FastCollector(actor, episodes, "cpu", pool=pool, check_every=7)
    assert collector.competitive and collector.n_actions == (38 if cfg["redouble"] else 37)
    seen = {"xx": 0, "fired": 0, "x": 0}
    for seed in (21, 22):
        fast = collector.collect(data, torch.Generator().manual_seed(seed), scorer, cfg["silent"])
        old = teacher_forced_old_competitive(actor, data, episodes,
                                             torch.Generator().manual_seed(seed), scorer,
                                             cfg["silent"], pool)
        assert isinstance(fast, CompetitiveTrajectories)
        assert type(fast.states) is type(old.states) and type(fast.terminal) is type(old.terminal)
        assert_same(fast, old)
        for name in ("sac_fired", "sac_credit", "clean", "standing_xx", "xx_delta"):
            assert torch.equal(getattr(fast, name), getattr(old, name)), name
        seen["x"] += int((fast.actions == DOUBLE).sum())
        seen["xx"] += int((fast.actions == REDOUBLE).sum())
        seen["fired"] += int(fast.sac_fired.sum())
        for kwargs in (dict(), dict(gate_pg=True, all_spots=True)):
            a = competitive_trajectory_losses(actor, critic, data, fast, **kwargs)
            b = competitive_trajectory_losses(actor, critic, data, old, **kwargs)
            assert set(a) == set(b)
            for key in b:
                assert torch.equal(a[key], b[key]), key
    assert seen["x"] > 0 and (seen["xx"] > 0) == cfg["redouble"]
    assert (seen["fired"] > 0) == cfg["sacrifice"]
    assert not bool(fast.clean.all())                    # sampled non-greedy opponent calls


def test_inverse_cdf_sample_has_the_policy_distribution():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(1, 37, generator=gen) * 2
    legal = torch.rand(1, 37, generator=gen) < 0.5
    legal[0, 35] = True
    out = {"policy_logits": logits, "double_gate": logits[:, 36]}
    log_probs = policy_log_probs(out, legal).expand(200000, -1)
    u = torch.rand(200000, generator=gen)
    u[:5] = 1.0 - 1e-7                                    # CDF edge never yields an illegal call
    sample = inverse_cdf_sample(log_probs, u, legal.expand(200000, -1))
    assert bool(legal[0, sample].all())
    freq = torch.bincount(sample, minlength=37).float() / len(sample)
    p = log_probs[0].exp()
    assert float((freq - p).abs().max()) < 4 * float((p * (1 - p) / len(sample)).sqrt().max()) + 1e-3
