import numpy as np
import pytest

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import DOUBLE, PASS, REDOUBLE, parse_call


def calls(text):
    return [parse_call(x) for x in text.split()]


def test_passout():
    state = AuctionState.from_calls([PASS] * 4)
    assert state.ended
    assert state.passed_out
    assert state.declarer() == -1


def test_three_passes_end_contract():
    state = AuctionState.from_calls(calls("1C P P P"))
    assert state.ended
    assert state.final_contract() == "1C by N"


def test_double_and_redouble_legality():
    state = AuctionState.from_calls(calls("1H"))
    assert state.legal_mask()[DOUBLE]
    assert not state.legal_mask()[REDOUBLE]
    state = state.apply(DOUBLE)
    assert state.legal_mask()[REDOUBLE]
    assert not state.legal_mask()[DOUBLE]
    state = state.apply(REDOUBLE)
    assert not state.legal_mask()[DOUBLE]
    assert not state.legal_mask()[REDOUBLE]
    state = AuctionState.from_calls(calls("1H P"))
    assert not state.legal_mask()[DOUBLE]  # partner cannot double own contract


def test_new_contract_clears_double():
    state = AuctionState.from_calls(calls("1H X 1S"))
    assert state.doubled == 0
    assert state.contract_seat == 2
    assert state.legal_mask()[DOUBLE]


def test_declarer_is_first_to_name_final_strain():
    state = AuctionState.from_calls(calls("1H P 2C P 2H P 4H P P P"))
    assert state.declarer() == 0
    assert state.final_contract() == "4H by N"


def test_illegal_call_rejected():
    state = AuctionState.from_calls(calls("1S"))
    with pytest.raises(ValueError):
        state.apply(parse_call("1H"))


def test_ended_has_no_legal_calls():
    state = AuctionState.from_calls(calls("1NT P P P"))
    assert not np.any(state.legal_mask())

