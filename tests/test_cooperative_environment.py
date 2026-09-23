"""Stage D1 invariants: silent opponents, no X/XX, endpoint targets, no leakage."""

from pathlib import Path

import numpy as np
import pytest
import torch

from bridgezero.bridge.calls import DOUBLE, PASS, REDOUBLE, parse_call
from bridgezero.bridge.deals import load_dataset
from bridgezero.bridge.scoring import contract_score
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.environment import CooperativeAuction
from bridgezero.contract.model import AuctionContractNet
from bridgezero.contract.prefixes import (
    CoopBatch,
    exact_endpoint,
    final_scores,
    sample_prefixes,
)
from bridgezero.contract.targets import TorchScorer
from bridgezero.contract.train_auction import check_invariants, d1_losses

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def smoke():
    owners, tricks = load_dataset(SMOKE)
    return owners, tricks, TorchDeals(owners, tricks)


def calls(text):
    return [parse_call(x) for x in text.split()]


def random_reference_game(rng, max_calls=8):
    game = CooperativeAuction.new(int(rng.integers(4)), int(rng.integers(2)),
                                  bool(rng.integers(2)))
    for _ in range(int(rng.integers(0, max_calls))):
        if game.ended:
            break
        legal = np.flatnonzero(game.legal_mask())
        game = game.apply(int(rng.choice(legal)))
    return game


# --- inactive opponents always Pass ---------------------------------------


def test_inactive_seats_can_only_pass_and_are_advanced_automatically():
    game = CooperativeAuction.new(dealer=1, active_side=0, vulnerable=False)
    assert game.actor == 2                   # East's forced Pass was already applied
    assert game.state.calls == (PASS,)
    raw = game.state.apply(parse_call("1H"))  # West (inactive) to act in raw state
    inactive = CooperativeAuction(raw, 0)
    mask = inactive.legal_mask()
    assert mask[PASS] and mask.sum() == 1
    with pytest.raises(ValueError):
        inactive.apply(parse_call("2C"))


def test_random_reference_auctions_never_let_opponents_bid():
    rng = np.random.default_rng(0)
    for _ in range(500):
        game = random_reference_game(rng, 20)
        assert game.inactive_calls_are_passes()
        assert all(call not in (DOUBLE, REDOUBLE) for call in game.state.calls)


def test_sampled_prefixes_replay_through_reference_wrapper():
    _, _, deals = smoke()
    gen = torch.Generator().manual_seed(1)
    net = AuctionContractNet(width=32, suit_width=8, depth=1)
    batch, info = sample_prefixes(deals, 512, gen, online=net, target=net)
    assert check_invariants(batch, deals, n=512) == 512
    assert set(info["depth"].tolist()) >= {0, 1, 2, 3}


# --- X/XX never enter masks, targets, or replay ------------------------------


def test_double_and_redouble_absent_from_masks_and_targets():
    owners, tricks, deals = smoke()
    # A state where the four-seat engine would allow an opponent Double.
    raw = CooperativeAuction.new(0, 0, False).state.apply(parse_call("1S"))
    assert raw.legal_mask()[DOUBLE]                 # raw engine: East could double
    east = CooperativeAuction(raw, 0)
    assert not east.legal_mask()[DOUBLE] and not east.legal_mask()[REDOUBLE]
    with pytest.raises(ValueError):
        east.apply(DOUBLE)
    game = east._advance_inactive()                 # South to act
    assert not game.legal_mask()[36:].any()
    with pytest.raises(ValueError):
        game.apply(REDOUBLE)
    gen = torch.Generator().manual_seed(2)
    batch, _ = sample_prefixes(deals, 256, gen)
    values, _, _ = exact_endpoint(batch, deals, TorchScorer())
    assert batch.legal().shape[1] == 36 and values.shape[1] == 36
    assert int(batch.history.max()) <= PASS
    net = AuctionContractNet(width=32, suit_width=8, depth=1)
    out = net(deals.hands[batch.deal, batch.actor_seat], batch.features())
    assert out["policy_logits"].shape[1] == 36 and out["contract_q"].shape[1] == 36
    losses = d1_losses(net, net, deals, batch, TorchScorer(), 0.5)
    assert all(torch.isfinite(v) for v in losses.values())


# --- Pass preserves the standing partnership contract -----------------------


def test_pass_with_no_bid_scores_zero():
    tricks = np.full((4, 5), 13)
    game = CooperativeAuction.new(0, 0, True)
    assert game.endpoint_score(PASS, tricks) == 0


def test_pass_after_partner_bid_keeps_partner_contract_and_declarer():
    tricks = np.zeros((4, 5), dtype=np.int64)
    tricks[0, 1] = 10    # North takes 10 in hearts
    tricks[2, 1] = 6     # South only 6
    game = CooperativeAuction.new(0, 0, False).apply(parse_call("4H"))
    assert game.actor == 2
    assert game.endpoint_score(PASS, tricks) == 420            # North declares 4H
    assert game.endpoint_score(parse_call("5H"), tricks) == -50  # still North declares


def test_higher_bid_replaces_contract_and_new_strain_declared_by_actor():
    tricks = np.zeros((4, 5), dtype=np.int64)
    tricks[2, 4] = 9     # South makes 3NT
    tricks[0, 4] = 5
    game = CooperativeAuction.new(0, 0, True).apply(parse_call("1H"))
    assert game.endpoint_score(parse_call("3NT"), tricks) == 600   # South names NT
    # Raising hearts keeps North (first to name hearts) as declarer: 10 down vul.
    assert game.endpoint_score(parse_call("4H"), tricks) == contract_score(4, 2, 0, 0, True)
    game = game.apply(parse_call("1NT")).apply(parse_call("2H"))  # S, then N
    assert game.actor == 2
    assert game.endpoint_score(PASS, tricks) == contract_score(2, 2, 0, 0, True)
    # South bid NT first, so South still declares 3NT.
    assert game.endpoint_score(parse_call("3NT"), tricks) == 600


def test_batch_endpoint_targets_match_reference_for_every_legal_call():
    owners, tricks, deals = smoke()
    gen = torch.Generator().manual_seed(3)
    batch, _ = sample_prefixes(deals, 200, gen, window=35)
    values, ceiling, _ = exact_endpoint(batch, deals, TorchScorer())
    for row, history in enumerate(batch.call_lists()):
        game = CooperativeAuction.new(int(batch.dealer[row]), int(batch.side[row]),
                                      bool(batch.vul[row]))
        for call in history:
            game = game.apply(call)
        table = tricks[int(batch.deal[row])]
        for action in np.flatnonzero(game.legal_mask()):
            assert values[row, action] == game.endpoint_score(int(action), table)
        assert values[row].max() <= ceiling[row]


def test_final_scores_match_reference_terminal_scoring():
    owners, tricks, deals = smoke()
    rng = np.random.default_rng(4)
    n = 300
    deal = torch.as_tensor(rng.integers(0, len(owners), n))
    side = torch.as_tensor(rng.integers(0, 2, n))
    dealer = torch.as_tensor(rng.integers(0, 4, n))
    vul = torch.as_tensor(rng.integers(0, 2, n))
    batch = CoopBatch.start(deal, side, dealer, vul)
    while not bool(batch.ended.all()):
        legal = batch.legal().float() + 1e-9
        action = torch.multinomial(legal, 1).squeeze(1)
        batch.apply(action, ~batch.ended)
    score, _, _ = final_scores(batch, deals, TorchScorer())
    for row, history in enumerate(batch.call_lists()):
        game = CooperativeAuction.new(int(dealer[row]), int(side[row]), bool(vul[row]))
        for call in history:
            game = game.apply(call)
        assert game.ended and game.inactive_calls_are_passes()
        assert score[row] == game.partnership_score(game.state, tricks[int(deal[row])])


# --- observations contain no hidden information -----------------------------


def test_observation_ignores_unseen_cards_and_dds_labels():
    owners, tricks, deals = smoke()
    rng = np.random.default_rng(5)
    gen = torch.Generator().manual_seed(6)
    batch, _ = sample_prefixes(deals, 128, gen)
    actor = batch.actor_seat.numpy()
    deal = batch.deal.numpy()
    # Scrambled DDS labels must not change any network output.
    other = TorchDeals(owners, rng.integers(0, 14, tricks.shape))
    net = AuctionContractNet(width=32, suit_width=8, depth=1).eval()
    a = net(deals.hands[batch.deal, batch.actor_seat], batch.features())
    b = net(other.hands[batch.deal, batch.actor_seat], batch.features())
    for key in a:
        assert torch.equal(a[key], b[key])
    # Permuting cards the actor cannot see leaves the observation unchanged.
    for row in range(32):
        d, s = int(deal[row]), int(actor[row])
        hidden = np.flatnonzero(owners[d] != s)
        changed = owners[d].copy()
        changed[hidden] = rng.permutation(owners[d, hidden])
        game = CooperativeAuction.new(int(batch.dealer[row]), int(batch.side[row]),
                                      bool(batch.vul[row]))
        for call in batch.call_lists()[row]:
            game = game.apply(call)
        hand_a, feat_a = game.observation(owners[d])
        hand_b, feat_b = game.observation(changed)
        assert np.array_equal(hand_a, hand_b) and np.array_equal(feat_a, feat_b)
        assert np.array_equal(feat_a, batch.features()[row].numpy())
        assert np.array_equal(hand_a, deals.hands[d, s].numpy())


def test_observation_is_rotation_invariant():
    owners, tricks, deals = smoke()
    rotated = TorchDeals((owners + 1) % 4, np.roll(tricks, 1, axis=1))
    gen = torch.Generator().manual_seed(7)
    batch, _ = sample_prefixes(deals, 256, gen)
    turned = CoopBatch(**{**batch.__dict__, "dealer": (batch.dealer + 1) % 4,
                          "side": 1 - batch.side})
    assert torch.equal(turned.actor_seat, (batch.actor_seat + 1) % 4)
    assert torch.equal(batch.features(), turned.features())
    assert torch.equal(deals.hands[batch.deal, batch.actor_seat],
                       rotated.hands[turned.deal, turned.actor_seat])
    scorer = TorchScorer()
    assert torch.equal(exact_endpoint(batch, deals, scorer)[0],
                       exact_endpoint(turned, rotated, scorer)[0])


def test_d1_trainer_end_to_end_smoke(tmp_path):
    from bridgezero.contract.train_auction import parse_args, run
    args = parse_args([
        "--data", str(SMOKE), "--out", str(tmp_path), "--train-count", "96",
        "--val-start", "96", "--val-count", "16", "--eval-start", "-16", "--eval-count", "16",
        "--steps", "4", "--batch", "64", "--eval-every", "2", "--target-every", "2",
        "--width", "16", "--suit-width", "8", "--depth", "1", "--threads", "1"])
    report = run(args)
    ckpt = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ckpt["stage"] == "D1" and ckpt["model_kind"] == "auction"
    assert report["rows"] == 16 * 16
    for rule in report["rules"].values():
        assert rule["mean_regret"] >= 0
