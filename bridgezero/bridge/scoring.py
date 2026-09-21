"""Duplicate bridge scoring, IMP conversion, and double-dummy par."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

from .auction import AuctionState
from .calls import CONTRACTS, PASS


def contract_score(
    level: int,
    strain_idx: int,
    tricks: int,
    doubled: int = 0,
    vulnerable: bool = False,
) -> int:
    """Score from declarer's perspective; doubled is 0, 1 (X), or 2 (XX)."""
    if doubled not in (0, 1, 2):
        raise ValueError("doubled must be 0, 1, or 2")
    needed = level + 6
    if tricks < needed:
        down = needed - tricks
        if doubled == 0:
            return -(100 if vulnerable else 50) * down
        if vulnerable:
            penalty = 200 + 300 * (down - 1)
        elif down == 1:
            penalty = 100
        elif down <= 3:
            penalty = 100 + 200 * (down - 1)
        else:
            penalty = 500 + 300 * (down - 3)
        return -penalty * (2 if doubled == 2 else 1)

    per_trick = 20 if strain_idx <= 1 else 30
    base = per_trick * level + (10 if strain_idx == 4 else 0)
    multiplier = (1, 2, 4)[doubled]
    contract_points = base * multiplier
    score = contract_points
    if doubled == 1:
        score += 50
    elif doubled == 2:
        score += 100
    score += (500 if vulnerable else 300) if contract_points >= 100 else 50

    over = tricks - needed
    if doubled == 0:
        score += over * per_trick
    else:
        score += over * (200 if vulnerable else 100) * (2 if doubled == 2 else 1)
    if level == 6:
        score += 750 if vulnerable else 500
    elif level == 7:
        score += 1500 if vulnerable else 1000
    return score


def terminal_ns_score(state: AuctionState, tricks: np.ndarray) -> int:
    """Score a finished auction using tricks[seat, strain] in S,H,D,C,NT order."""
    if not state.ended:
        raise ValueError("auction is not over")
    if state.last_contract < 0:
        return 0
    _, level, bid_strain = CONTRACTS[state.last_contract]
    # Bidding order is C,D,H,S,NT; trick table order is S,H,D,C,NT.
    trick_strain = (3, 2, 1, 0, 4)[bid_strain]
    declarer = state.declarer()
    raw = contract_score(level, bid_strain, int(tricks[declarer, trick_strain]),
                         state.doubled, state.vulnerable(declarer % 2))
    return raw if declarer % 2 == 0 else -raw


def stand_pat_actor_score(state: AuctionState, action: int,
                          tricks: np.ndarray) -> int:
    """Actor-perspective score if ``action`` is followed only by Passes.

    This is a diagnostic, not a shaped reward.  In particular, comparing X to
    Pass at the same state cleanly measures whether the double itself pays when
    neither partnership changes the contract afterwards.
    """
    side = state.side_to_act
    final = state.apply(action)
    while not final.ended:
        final = final.apply(PASS)
    ns_score = terminal_ns_score(final, tricks)
    return ns_score if side == 0 else -ns_score


IMP_LOWER_BOUNDS = (20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500,
                    600, 750, 900, 1100, 1300, 1500, 1750, 2000, 2250,
                    2500, 3000, 3500, 4000)


def imps(score_difference: int | float) -> int:
    sign = 1 if score_difference >= 0 else -1
    return sign * bisect_right(IMP_LOWER_BOUNDS, abs(score_difference))


def dd_par_score(tricks: np.ndarray, vul_ns: bool, vul_ew: bool) -> int:
    """Double-dummy ladder minimax, returned as an NS score.

    This uses the best declarer for each side/strain and permits the defenders to
    double. Redouble never improves the minimax result because a rational defender
    does not double a contract whose redouble would benefit declarer.
    """
    n_contracts = len(CONTRACTS)
    stand_ns = np.empty(n_contracts, dtype=np.int32)
    stand_ew = np.empty(n_contracts, dtype=np.int32)
    trick_perm = (3, 2, 1, 0, 4)
    for action, (_, level, bid_strain) in enumerate(CONTRACTS):
        ts = trick_perm[bid_strain]
        best_ns = []
        best_ew = []
        for doubled in (0, 1):
            best_ns.append(max(contract_score(level, bid_strain, int(tricks[s, ts]),
                                              doubled, vul_ns) for s in (0, 2)))
            best_ew.append(max(contract_score(level, bid_strain, int(tricks[s, ts]),
                                              doubled, vul_ew) for s in (1, 3)))
        stand_ns[action] = min(best_ns)
        stand_ew[action] = -min(best_ew)

    a = np.empty(n_contracts, dtype=np.int32)
    b = np.empty(n_contracts, dtype=np.int32)
    min_b = 10**9
    max_a = -10**9
    for k in range(n_contracts - 1, -1, -1):
        a[k] = min(int(stand_ns[k]), min_b)
        b[k] = max(int(stand_ew[k]), max_a)
        min_b = min(min_b, int(b[k]))
        max_a = max(max_a, int(a[k]))
    return max(min(0, min_b), max_a)


def dd_cooperative_score(tricks: np.ndarray, side: int, vulnerable: bool) -> int:
    """Best undoubled score a partnership can reach with opponents passing.

    Returned from that partnership's perspective and clamped at zero because the
    partnership may pass the deal out.
    """
    trick_perm = (3, 2, 1, 0, 4)
    best = 0
    for action, (_, level, bid_strain) in enumerate(CONTRACTS):
        del action
        table_strain = trick_perm[bid_strain]
        for declarer in (side, side + 2):
            best = max(best, contract_score(
                level, bid_strain, int(tricks[declarer, table_strain]),
                doubled=0, vulnerable=vulnerable))
    return best


def own_contract_score(state: AuctionState, tricks: np.ndarray, side: int) -> int:
    """Score a side's own highest bid as an undoubled contract.

    Used only by the bootstrap communication curriculum. It supplies a dense
    cooperative signal even when the opponents win the real auction.
    """
    own_calls = []
    for i, action in enumerate(state.calls):
        seat = (state.dealer + i) % 4
        if action < 35 and seat % 2 == side:
            own_calls.append((action, seat))
    if not own_calls:
        return 0
    contract = max(action for action, _ in own_calls)
    _, level, strain = CONTRACTS[contract]
    declarer = next(
        seat for action, seat in own_calls if CONTRACTS[action][2] == strain)
    table_strain = (3, 2, 1, 0, 4)[strain]
    return contract_score(level, strain, int(tricks[declarer, table_strain]),
                          doubled=0, vulnerable=state.vulnerable(side))
