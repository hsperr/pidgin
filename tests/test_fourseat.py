from pathlib import Path

import numpy as np
import pytest
import torch

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import PASS
from bridgezero.bridge.deals import load_dataset
from bridgezero.bridge.scoring import own_contract_score
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.environment import AUCTION_FEATURES
from bridgezero.contract.model import AuctionContractNet, CentralCritic, save_checkpoint
from bridgezero.contract.prefixes import CoopBatch
from bridgezero.contract.targets import TorchScorer
from bridgezero.fourseat.model import (
    FourSeatNet,
    SilentView,
    warm_start,
)
from bridgezero.fourseat.rollout import collect_trajectories
from bridgezero.fourseat.state import (
    FOURSEAT_FEATURES,
    FourSeatBatch,
    features_from_history,
    own_bid_scores,
)

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def random_batch(n, seed, silent_frac=0.0, bid_prob=0.45):
    """Random legal four-seat auctions, also replayed through AuctionState."""
    gen = torch.Generator().manual_seed(seed)
    deal = torch.randint(128, (n,), generator=gen)
    dealer = torch.randint(4, (n,), generator=gen)
    vns, vew = torch.randint(2, (n,), generator=gen), torch.randint(2, (n,), generator=gen)
    silent = torch.where(torch.rand(n, generator=gen) < silent_frac,
                         torch.randint(2, (n,), generator=gen), torch.full((n,), -1))
    batch = FourSeatBatch.start(deal, dealer, vns, vew, silent)
    refs = [AuctionState(dealer=int(dealer[i]), vul_ns=bool(vns[i]), vul_ew=bool(vew[i]))
            for i in range(n)]
    snapshots = []
    while not bool(batch.ended.all()):
        legal = batch.legal()
        for i in range(n):
            if refs[i].ended:
                assert not legal[i].any()
                continue
            expect = refs[i].legal_mask()[:36].copy()
            if int(silent[i]) == refs[i].side_to_act:
                expect[:35] = False
            assert np.array_equal(legal[i].numpy(), expect)
        snapshots.append((batch.subset(torch.arange(n)), [r for r in refs]))
        action = torch.full((n,), PASS)
        for i in range(n):
            options = legal[i, :35].nonzero().squeeze(1)
            if len(options) and torch.rand(1, generator=gen) < bid_prob:
                action[i] = options[torch.randint(min(len(options), 6), (1,), generator=gen)]
        batch.apply(action, ~batch.ended)
        for i in range(n):
            if not refs[i].ended:
                refs[i] = refs[i].apply(int(action[i]))
        assert [r.ended for r in refs] == batch.ended.tolist()
    return batch, refs, snapshots


def slow_features(state: AuctionState, actor: int) -> np.ndarray:
    f = np.zeros(FOURSEAT_FEATURES, dtype=np.float32)
    seen = False
    for i, call in enumerate(state.calls):
        seat = (state.dealer + i) % 4
        rel = (seat - actor) % 4
        if call < PASS:
            f[{0: 0, 2: 35, 1: 77, 3: 112}[rel] + call] = 1
            seen |= rel % 2 == 0
        elif rel % 2 == 0 and not seen:
            f[70 if rel == 0 else 71] = 1
    f[72 + (state.dealer - actor) % 4] = 1
    f[76] = float(state.vulnerable(actor % 2))
    return f


def test_legality_termination_and_no_double():
    batch, refs, _ = random_batch(64, 1, silent_frac=0.3)
    for i, ref in enumerate(refs):
        assert batch.call_lists()[i] == list(ref.calls)
    assert batch.legal().shape[1] == 36


def test_opponent_bid_raises_the_ladder():
    b = FourSeatBatch.start(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]),
                            torch.tensor([0]))
    b.apply(torch.tensor([14]), torch.tensor([True]))   # N 3NT
    legal = b.legal()                                  # E to act
    assert not legal[0, :15].any() and legal[0, 15:].all()
    b.apply(torch.tensor([PASS]), torch.tensor([True]))
    with pytest.raises(ValueError):
        b.apply(torch.tensor([13]), torch.tensor([True]))  # S cannot bid 3S over 3NT
    with pytest.raises(ValueError):
        b.apply(torch.tensor([36]), torch.tensor([True]))  # no Double slot


def test_own_bid_reward_matches_reference_scoring():
    data = deals()
    batch, refs, _ = random_batch(128, 2)
    score, ceiling, table_ns = own_bid_scores(batch, data, TorchScorer())
    tricks = data.tricks.numpy()
    from bridgezero.bridge.scoring import dd_cooperative_score, terminal_ns_score
    for i, ref in enumerate(refs):
        t = tricks[int(batch.deal[i])]
        for side in (0, 1):
            assert score[i, side] == own_contract_score(ref, t, side)
            assert ceiling[i, side] == dd_cooperative_score(t, side, ref.vulnerable(side))
        assert table_ns[i] == terminal_ns_score(ref, t)


def test_features_match_slow_reference():
    _, _, snaps = random_batch(48, 3, silent_frac=0.2)
    for batch, refs in snaps:
        feats = batch.features()
        for i, ref in enumerate(refs):
            if ref.ended:
                continue
            assert np.array_equal(feats[i].numpy(), slow_features(ref, ref.turn))
        hist = batch.history[:, :int(batch.t.max())]
        again = features_from_history(hist, batch.dealer, batch.vul[:, 0], batch.vul[:, 1],
                                      batch.actor_seat)
        assert torch.equal(again, feats)


def silent_pair(n, seed):
    """The same random silent auctions as CoopBatch and FourSeatBatch states."""
    gen = torch.Generator().manual_seed(seed)
    deal, side = torch.randint(128, (n,), generator=gen), torch.randint(2, (n,), generator=gen)
    dealer, vul = torch.randint(4, (n,), generator=gen), torch.randint(2, (n,), generator=gen)
    coop = CoopBatch.start(deal, side, dealer, vul)
    four = FourSeatBatch.start(deal, dealer, vul * (side == 0), vul * (side == 1), 1 - side)
    coops, fours = [], []
    while not bool(coop.ended.all()):
        alive = ~coop.ended
        while bool(four.forced.any()):
            four.apply(torch.full((n,), PASS), four.forced)
        assert torch.equal(four.actor_seat[alive], coop.actor_seat[alive])
        ids = alive.nonzero().squeeze(1)
        coops.append(coop.subset(ids))
        fours.append(four.subset(ids))
        legal = coop.legal()
        assert torch.equal(legal[ids], four.legal()[ids])
        probs = legal.float() * torch.rand(n, 36, generator=gen)
        probs[:, 35] += 0.5
        action = probs.argmax(1)
        coop.apply(action, alive)
        four.apply(action, alive)
    return CoopBatch.cat(coops), FourSeatBatch.cat(fours)


def test_silent_features_extend_cooperative_features():
    coop, four = silent_pair(96, 4)
    feats = four.features()
    assert torch.equal(feats[:, :AUCTION_FEATURES], coop.features())
    assert not feats[:, AUCTION_FEATURES:].any()


def test_warm_start_is_bit_identical_on_silent_states(tmp_path):
    torch.manual_seed(0)
    base = AuctionContractNet(32, 8, 2)
    critic = CentralCritic(32, 8, 2)
    with torch.no_grad():
        for p in (*base.parameters(), *critic.parameters()):
            p.normal_(0, 0.3)
    path = tmp_path / "coop.pt"
    save_checkpoint(path, base, "D4PG", step=1, critic_config=critic.config,
                    critic=critic.state_dict())
    actor4, critic4, meta = warm_start(path)
    assert len(meta["init_sha256"]) == 64
    data = deals()
    coop, four = silent_pair(128, 5)
    hand = data.hands[coop.deal, coop.actor_seat]
    old, new = base(hand, coop.features()), actor4(hand, four.features())
    for key in old:
        assert torch.equal(old[key], new[key]), key
    assert torch.equal(SilentView(actor4)(hand, coop.features())["policy_logits"],
                       old["policy_logits"])
    pair = torch.stack((hand, data.hands[coop.deal, (coop.actor_seat + 2) % 4]), 1)
    assert torch.equal(critic(pair, coop.features()), critic4(pair, four.features()))
    # Opponent bids do change the new network once its opponent weights move.
    with torch.no_grad():
        actor4.auction_net[0].opponent.weight.normal_()
    opp = four.features().clone()
    opp[:, 80] = 1
    assert not torch.equal(actor4(hand, opp)["policy_logits"], old["policy_logits"])


def test_trajectories_group_returns_by_side():
    data = deals()
    actor = FourSeatNet(16, 8, 1)
    traj = collect_trajectories(actor, data, 32, torch.Generator().manual_seed(6), TorchScorer(),
                                silent_frac=0.5)
    assert bool(traj.terminal.ended.all())
    for g in range(len(traj.score)):
        sides = traj.states.side[traj.episode == g]
        assert (sides == sides[0]).all()
    # Silent seats never make recorded decisions.
    assert not bool((traj.states.side == traj.states.silent).any())
    assert bool((traj.terminal.silent >= 0).any())
