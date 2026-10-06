"""Rule bidders: legal, finished auctions, and openings that match their descriptions."""

from pathlib import Path

import pytest
import torch

from training.bridge.auction import AuctionState
from training.bridge.calls import PASS
from training.bridge.deals import load_dataset
from training.fourseat.rulebots import STYLES, RuleBot

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"
POINTS = torch.tensor(([4, 3, 2, 1] + [0] * 9) * 4, dtype=torch.float32)


def hands_of(owners):
    owners = torch.as_tensor(owners)
    return torch.stack([(owners == s).float() for s in range(4)], 1)       # (B,4,52)


def play(bots, owners, dealer):
    """bots[side]; returns the finished call lists."""
    hands = hands_of(owners)
    B, H = len(dealer), 80
    history = torch.full((B, H), -1, dtype=torch.long)
    states = [AuctionState(dealer=int(d)) for d in dealer]
    for t in range(H):
        alive = torch.tensor([not s.ended for s in states])
        if not alive.any():
            break
        seat = (dealer + t) % 4
        calls = torch.full((B,), PASS, dtype=torch.long)
        for side in (0, 1):
            rows = alive & (seat % 2 == side)
            if rows.any():
                idx = rows.nonzero().squeeze(1)
                calls[idx] = bots[side].act(hands[idx, seat[idx]], history[idx],
                                            torch.full((len(idx),), t), dealer[idx])
        for i in alive.nonzero().squeeze(1).tolist():
            states[i] = states[i].apply(int(calls[i]))       # raises on an illegal call
            history[i, t] = calls[i]
    assert all(s.ended for s in states)
    return states


@pytest.mark.parametrize("style", STYLES)
def test_rule_bots_bid_legal_finished_auctions(style):
    owners, _ = load_dataset(SMOKE)
    dealer = torch.arange(len(owners)) % 4
    states = play([RuleBot(style), RuleBot("sayc")], owners, dealer)
    opened = sum(1 for s in states if s.last_contract >= 0)
    assert opened > len(states) * 0.5


def test_openings_follow_the_system():
    owners, _ = load_dataset(SMOKE)
    hands = hands_of(owners)[:, 0]                                         # North deals
    hcp = (hands * POINTS).sum(1)
    empty = torch.full((len(hands), 4), -1, dtype=torch.long)
    zero = torch.zeros(len(hands), dtype=torch.long)
    sayc = RuleBot("sayc").act(hands, empty, zero, zero)
    weak = RuleBot("weakclub").act(hands, empty, zero, zero)
    assert (hcp[sayc == PASS] <= 11).all()
    assert (hcp[sayc == 4] >= 15).all() and (hcp[sayc == 4] <= 17).all()      # 1NT
    assert (weak[hcp <= 10] == 0).all()                                         # weak 1C
    assert (weak != PASS).all()                                                 # opens every hand
