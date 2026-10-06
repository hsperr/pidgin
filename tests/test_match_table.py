"""tools/match.py: the table holds the longest legal auction with X and XX."""

import importlib.util
from pathlib import Path

import torch

from training.bridge.auction import AuctionState
from training.bridge.calls import DOUBLE, PASS, REDOUBLE
from training.bridge.deals import load_dataset
from training.contract.data import TorchDeals
from training.fourseat.competitive import MAX_REDOUBLE_CALLS

ROOT = Path(__file__).resolve().parents[1]


def longest_auction() -> list[int]:
    """3 passes, then every bid followed by P P X P P XX P P; the last one ends P P P (319)."""
    calls = [PASS] * 3
    between = [PASS, PASS, DOUBLE, PASS, PASS, REDOUBLE, PASS, PASS]
    for bid in range(35):
        calls += [bid, *between]
    return calls + [PASS]


def test_match_table_plays_longest_legal_auction():
    spec = importlib.util.spec_from_file_location("match_table", ROOT / "tools" / "match.py")
    match = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(match)
    calls = longest_auction()
    assert len(calls) == 319 <= match.MAX_TABLE_CALLS == MAX_REDOUBLE_CALLS
    state = AuctionState.from_calls(calls, dealer=0, vul_ns=False, vul_ew=False)
    assert state.ended and state.last_contract == 34 and state.doubled == 2

    class Scripted:
        def act(self, table, rows):
            return torch.full((len(rows),), calls[table.t], dtype=torch.long)

    deals = TorchDeals(*load_dataset(ROOT / "data" / "smoke_128.npz"))
    n = 4
    table = match.play([Scripted(), Scripted()], deals, torch.arange(n), torch.zeros(n, dtype=torch.long),
                       torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
                       torch.tensor([[0, 1]]).expand(n, 2))
    assert table.t == 319 and not bool(table.st.alive.any())
    assert table.history[:, :319].tolist() == [calls] * n
    ns, contract, _ = table.ns_scores()
    assert (contract == 34).all()
