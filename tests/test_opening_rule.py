"""Opening rule (rule of 18): in 1st/2nd seat a 1-level opening needs HCP + two longest suits >= 18."""

import pytest
import torch

from bridgezero.bridge.calls import PASS
from bridgezero.contract.targets import TorchScorer
from bridgezero.fourseat import competitive
from bridgezero.fourseat.competitive import (
    ONE_LEVEL,
    apply_opening_rule,
    competitive_trajectory_losses,
    opening_points,
    set_opening_rule,
)
from bridgezero.fourseat.fast_rollout import FastCollector
from bridgezero.fourseat.model import FourSeatCompetitiveCritic, FourSeatNet
from test_fast_rollout import competitive_actor, deals, noisy


def hand(cards: dict) -> torch.Tensor:
    """``{"S": "AKQ..", ...}`` -> ``(52,)``, suit * 13 + rank (A first)."""
    h = torch.zeros(52)
    for s, ranks in cards.items():
        for r in ranks:
            h["SHDC".index(s) * 13 + "AKQJT98765432".index(r)] = 1
    return h


@pytest.fixture(autouse=True)
def rule_off_after():
    yield
    set_opening_rule(0)


def test_opening_points():
    # 13 HCP + 5 + 4 = 22; 10 HCP + 4 + 3 = 17
    a = hand({"S": "AKJ32", "H": "Q432", "D": "K2", "C": "32"})
    b = hand({"S": "A432", "H": "K32", "D": "Q32", "C": "J32"})
    assert opening_points(torch.stack([a, b])).tolist() == [22.0, 17.0]


def test_mask_only_first_and_second_seat_before_any_bid():
    weak = hand({"S": "A432", "H": "K32", "D": "Q32", "C": "J32"}).expand(4, 52)
    legal = torch.ones(4, PASS + 1, dtype=torch.bool)
    last = torch.tensor([-1, -1, -1, 0])          # rows 0-2: no bid yet; row 3: 1C was bid
    t = torch.tensor([0, 1, 2, 1])                # row 2: after two passes (3rd seat)
    out = apply_opening_rule(legal, weak, last, t, 18)
    assert not out[0, :ONE_LEVEL].any() and not out[1, :ONE_LEVEL].any()
    assert out[0, ONE_LEVEL:].all()               # 2-level bids and Pass stay legal
    assert out[2].all() and out[3].all()
    assert torch.equal(apply_opening_rule(legal, weak, last, t, 0), legal)


def test_rollout_never_opens_light_in_first_two_seats_and_loss_is_finite():
    cfg = dict(redouble=True, sacrifice=True, bound=3.0)
    data = deals()
    actor = competitive_actor(cfg)
    with torch.no_grad():                         # push hard toward 1-level openings
        actor.policy_head.bias[:ONE_LEVEL] += 4.0
    critic = noisy(FourSeatCompetitiveCritic(24, 8, 2), 2)
    frozen = noisy(FourSeatNet(24, 8, 2), 3)
    with torch.no_grad():
        frozen.policy_head.bias[:ONE_LEVEL] += 4.0
    set_opening_rule(18)
    collector = FastCollector(actor, 128, "cpu", pool={1: (frozen, 0.5)}, check_every=7)
    traj = collector.collect(data, torch.Generator().manual_seed(5), TorchScorer(), 0.0)
    s = traj.states
    pts = opening_points(data.hands[s.deal, s.actor_seat])
    opener = s.last < 0
    blocked = opener & (s.t < 2) & (pts < 18)
    assert int(blocked.sum()) > 0
    assert not (traj.actions[blocked] < ONE_LEVEL).any()
    # every opening in the recorded auctions, the frozen opponent's included
    h = s.history
    first = torch.where(h >= 0, h, PASS).ne(PASS).float().argmax(1)
    call = h.gather(1, first[:, None]).squeeze(1)
    opened = (call >= 0) & (call < PASS)
    seat = (s.dealer + first) % 4
    opener_pts = opening_points(data.hands[s.deal, seat])
    early = opened & (first < 2) & (call < ONE_LEVEL)
    assert int(early.sum()) > 0
    assert (opener_pts[early] >= 18).all()
    assert competitive.OPENING_RULE == 18
    losses = competitive_trajectory_losses(actor, critic, data, traj)
    assert all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
