"""Duplicate scoring WITH doubles, for the four-seat game.

`scoring.py` scores an undoubled contract for declarer. That is all the
two-player experiments ever needed. The four-seat game needs the doubled
column as well, and it needs it to be exact, because the whole competitive
auction hangs off it: an undoubled sacrifice is nearly free (7NT down 13
non-vulnerable costs 650, and most sacrifices cost far less than the game
they steal), so without the double there is no penalty large enough to make
anyone stop bidding.

Non-vulnerable, doubled:

    making   2 x trick score, +50 insult, +300 if the DOUBLED trick score
             reaches 100 else +50 partscore, +100 per overtrick,
             +500 / +1000 for a small / grand slam
    down     100, 300, 500, then 300 more each  (so -800, -1100, ...)

Vulnerable doubled is here too, so `--vulnerable` is a one-line passthrough,
but no experiment uses it yet.

This module never changes `scoring.py`; `score4(..., doubled=0)` is checked
against `scoring.score_contract` in tests/test_exp10four.py.
"""
import numpy as np

from emergent.scoring import STRAINS


def score4(level, strain_idx, tricks, doubled=0, vulnerable=False):
    """Duplicate score for DECLARER's side. level=None (Pass) -> 0."""
    if level is None:
        return 0
    needed = 6 + level
    per = 20 if strain_idx <= 1 else 30
    base = per * level + (10 if strain_idx == 4 else 0)

    if tricks >= needed:
        over = tricks - needed
        if doubled:
            trick_score = 2 * base
            total = trick_score + 50                      # insult
            total += (500 if vulnerable else 300) if trick_score >= 100 else 50
            total += over * (200 if vulnerable else 100)
        else:
            trick_score = base
            total = trick_score
            total += (500 if vulnerable else 300) if trick_score >= 100 else 50
            total += over * per
        if level == 6:
            total += 750 if vulnerable else 500
        elif level == 7:
            total += 1500 if vulnerable else 1000
        return total

    down = needed - tricks
    if not doubled:
        return -(100 if vulnerable else 50) * down
    if vulnerable:
        # 200, then 300 each
        return -(200 + 300 * (down - 1))
    # 100, 200, 200, then 300 each -> 100, 300, 500, 800, 1100, ...
    if down == 1:
        return -100
    if down <= 3:
        return -(100 + 200 * (down - 1))
    return -(500 + 300 * (down - 3))


def build_score_table4(contracts, vulnerable=False):
    """(n_contracts, 2, 14): score for each contract, undoubled/doubled, by tricks."""
    tbl = np.zeros((len(contracts), 2, 14), dtype=np.float32)
    for ci, (_, lvl, si) in enumerate(contracts):
        for d in range(2):
            for t in range(14):
                tbl[ci, d, t] = score4(lvl, si, t, d, vulnerable)
    return tbl


def contract_levels(contracts):
    return np.array([0 if lvl is None else lvl for _, lvl, _ in contracts],
                    dtype=np.int64)


def contract_strains(contracts):
    """Our own strain index (0=C..4=NT) per contract; Pass -> -1."""
    return np.array([-1 if si is None else si for _, _, si in contracts],
                    dtype=np.int64)


assert STRAINS[4] == "NT"
