import numpy as np

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import parse_call
from bridgezero.bridge.scoring import (
    contract_score,
    imps,
    own_contract_score,
    stand_pat_actor_score,
    terminal_ns_score,
)


def test_common_made_scores():
    assert contract_score(3, 4, 9, 0, False) == 400  # 3NT non-vul
    assert contract_score(4, 3, 10, 0, True) == 620  # 4S vul
    assert contract_score(6, 2, 12, 0, False) == 980  # 6H non-vul
    assert contract_score(7, 4, 13, 0, True) == 2220  # 7NT vul


def test_doubled_and_redoubled_scores():
    assert contract_score(2, 2, 8, 1, False) == 470
    assert contract_score(2, 2, 9, 1, False) == 570
    assert contract_score(2, 2, 8, 2, False) == 640
    assert contract_score(4, 3, 8, 1, False) == -300
    assert contract_score(4, 3, 8, 2, False) == -600
    assert contract_score(3, 4, 7, 1, True) == -500


def test_imp_boundaries():
    assert imps(0) == 0
    assert imps(10) == 0
    assert imps(20) == 1
    assert imps(40) == 1
    assert imps(50) == 2
    assert imps(-420) == -9
    assert imps(4000) == 24


def test_terminal_declarer_and_sign():
    tricks = np.full((4, 5), 7, dtype=np.uint8)
    tricks[0, 1] = 10  # North makes 4H
    state = AuctionState.from_calls([parse_call(x) for x in "1H P 4H P P P".split()])
    assert terminal_ns_score(state, tricks) == 420
    tricks[1, 0] = 10  # East makes 4S
    state = AuctionState.from_calls([parse_call(x) for x in "P 1S P 4S P P P".split()])
    assert terminal_ns_score(state, tricks) == -420


def test_own_contract_score_ignores_opponents_final_bid():
    tricks = np.full((4, 5), 7, dtype=np.uint8)
    tricks[0, 1] = 10  # North makes 4H.
    tricks[1, 0] = 8   # East's 4S is two down.
    state = AuctionState.from_calls(
        [parse_call(x) for x in "1H 1S 4H 4S P P P".split()])
    assert own_contract_score(state, tricks, 0) == 420
    assert own_contract_score(state, tricks, 1) == -100


def test_stand_pat_double_edge_is_actor_relative():
    tricks = np.full((4, 5), 7, dtype=np.uint8)
    tricks[0, 0] = 5  # North is two down in 1S.
    state = AuctionState.from_calls([parse_call("1S")])
    pass_score = stand_pat_actor_score(state, parse_call("P"), tricks)
    double_score = stand_pat_actor_score(state, parse_call("X"), tricks)
    assert pass_score == 100       # East-West gain from defending 1S down two.
    assert double_score == 300
    assert double_score - pass_score == 200
