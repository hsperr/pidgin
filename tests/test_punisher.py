"""The DD punisher (``--punisher-frac``) doubles exactly the standing contracts that fail."""

import copy
from pathlib import Path

import pytest
import torch

from bridgezero.bridge.calls import DOUBLE, PASS
from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.targets import TorchScorer
from bridgezero.fourseat import competitive
from bridgezero.fourseat.fast_rollout import FastCollector
from bridgezero.fourseat.model import FourSeatCompetitiveNet

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"
TABLE_STRAIN = (3, 2, 1, 0, 4)              # bid strain C D H S NT -> tricks S H D C NT


def actor():
    torch.manual_seed(1)
    net = FourSeatCompetitiveNet(24, 8, 2, redouble=True, sacrifice=True)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
        net.policy_head.bias[PASS] += 2.0
    return net


@pytest.mark.parametrize("level", [1, 3])
def test_punisher_doubles_failing_contracts_only(level):
    deals = TorchDeals(*load_dataset(SMOKE))
    net = actor()
    punisher = copy.deepcopy(net).eval().requires_grad_(False)
    punisher.punisher_level, punisher.punisher_miss = level, 0.0
    competitive.set_any_seat_double(True)
    try:
        col = FastCollector(net, 256, "cpu", pool={2: (punisher, 1.0)})
        col.collect(deals, torch.Generator().manual_seed(0), TorchScorer(), 0.0)
    finally:
        competitive.set_any_seat_double(False)
    checked = doubles = 0
    for b in range(col.B):
        side, dealer = int(col.frozen_side[b]), int(col.dealer[b])
        last, cseat, doubled, first = -1, -1, False, {}
        for j, a in enumerate(int(x) for x in col.history[b] if x >= 0):
            seat = (dealer + j) % 4
            if seat % 2 == side and last >= 0 and not doubled and cseat % 2 != side:
                strain = last % 5
                decl = min((i, k) for (k, s), i in first.items()
                           if s == strain and k % 2 == cseat % 2)[1]
                tricks = int(deals.tricks[int(col.deal[b]), decl, TABLE_STRAIN[strain]])
                fails = tricks < last // 5 + 7 and last // 5 + 1 >= level
                assert (a == DOUBLE) == fails, (b, j)
                checked += 1
                doubles += a == DOUBLE
            if a < 35:
                first.setdefault((seat, a % 5), j)
                last, cseat, doubled = a, seat, False
            elif a == DOUBLE:
                doubled = True
    assert checked > 50 and doubles > 5
