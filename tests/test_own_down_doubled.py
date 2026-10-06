"""``--own-down-doubled``: a failing own contract costs that share of the doubled penalty."""

from pathlib import Path

import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.bridge.scoring import contract_score
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.targets import TorchScorer
from bridgezero.fourseat import state
from bridgezero.fourseat.fast_rollout import FastCollector
from bridgezero.fourseat.model import FourSeatCompetitiveNet
from bridgezero.bridge.calls import PASS

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"
TABLE_STRAIN = (3, 2, 1, 0, 4)


def test_failing_own_contracts_cost_a_share_of_the_doubled_set():
    deals, scorer = TorchDeals(*load_dataset(SMOKE)), TorchScorer()
    torch.manual_seed(3)
    net = FourSeatCompetitiveNet(24, 8, 2, redouble=True, sacrifice=True)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
        net.policy_head.bias[PASS] += 1.0
    terminal = FastCollector(net, 256, "cpu").collect(
        deals, torch.Generator().manual_seed(0), scorer, 0.25).terminal
    plain, ceiling, table = state.own_bid_scores(terminal, deals, scorer)
    try:
        state.set_own_down_doubled(0.4)
        shared, ceiling2, table2 = state.own_bid_scores(terminal, deals, scorer)
    finally:
        state.set_own_down_doubled(0.0)
    assert torch.equal(ceiling, ceiling2) and torch.equal(table, table2)
    seats = terminal.bid_seats()
    changed = 0
    for b in range(len(terminal)):
        for side in (0, 1):
            own = [(c, int(seats[b, c])) for c in range(35)
                   if seats[b, c] >= 0 and seats[b, c] % 2 == side]
            if not own:
                assert shared[b, side] == plain[b, side] == 0
                continue
            top = own[-1][0]
            first = min(c for c, s in own if c % 5 == top % 5)
            declarer = int(seats[b, first])
            tricks = int(deals.tricks[int(terminal.deal[b]), declarer, TABLE_STRAIN[top % 5]])
            vul = bool(terminal.vul[b, side])
            undoubled = contract_score(top // 5 + 1, top % 5, tricks, 0, vul)
            want = undoubled
            if tricks < top // 5 + 7:
                want = undoubled + 0.4 * (contract_score(top // 5 + 1, top % 5, tricks, 1, vul)
                                          - undoubled)
                changed += 1
            assert abs(float(plain[b, side]) - undoubled) < 1e-4
            assert abs(float(shared[b, side]) - want) < 1e-3
    assert changed > 20


def test_table_down_doubled_scores_every_failing_contract_doubled():
    deals = TorchDeals(*load_dataset(SMOKE))
    torch.manual_seed(5)
    net = FourSeatCompetitiveNet(24, 8, 2, redouble=True, sacrifice=True)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    terminal = FastCollector(net, 256, "cpu").collect(
        deals, torch.Generator().manual_seed(1), TorchScorer(), 0.25).terminal
    plain = state.table_ns_score(terminal, deals)
    try:
        state.set_table_down_doubled("both")
        perfect = state.table_ns_score(terminal, deals)
    finally:
        state.set_table_down_doubled("")
    declarer, undoubled, doubled = state.contract_results(terminal, deals)
    sign = torch.where(declarer % 2 == 0, 1.0, -1.0)
    down = (terminal.last >= 0) & (undoubled < 0)
    assert down.sum() > 10
    redoubled = terminal.redoubled
    assert torch.allclose(perfect[down & ~redoubled], (sign * doubled)[down & ~redoubled])
    assert torch.equal(perfect[~down | redoubled], plain[~down | redoubled])


def test_table_down_doubled_own_charges_only_the_declaring_side():
    deals = TorchDeals(*load_dataset(SMOKE))
    torch.manual_seed(5)
    net = FourSeatCompetitiveNet(24, 8, 2, redouble=True, sacrifice=True)
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.3)
    terminal = FastCollector(net, 256, "cpu").collect(
        deals, torch.Generator().manual_seed(1), TorchScorer(), 0.25).terminal
    real = state.table_ns_score(terminal, deals)
    perfect = state.table_ns_score(terminal, deals, perfect=True)
    try:
        state.set_table_down_doubled("own")
        sides = state.side_table_scores(terminal, deals)
    finally:
        state.set_table_down_doubled("")
    ns_declares = state.contract_declarer(terminal) % 2 == 0
    assert torch.equal(sides[:, 0], torch.where(ns_declares, perfect, real))
    assert torch.equal(sides[:, 1], -torch.where(ns_declares, real, perfect))
    down = (terminal.last >= 0) & (perfect != real)
    assert down.sum() > 10                        # not zero-sum on undoubled failing contracts
    assert ((sides[:, 0] + sides[:, 1])[down] < 0).all()
