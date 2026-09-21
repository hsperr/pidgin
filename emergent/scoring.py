"""Duplicate bridge scoring and the contract action space.

This is hard-coded rules knowledge (allowed by the plan). It says nothing
about which contract a hand should choose.
"""
import numpy as np

STRAINS = ["C", "D", "H", "S", "NT"]   # 0..4, matches trick-value ordering
STRAIN_TO_DD = {"S": 0, "H": 1, "D": 2, "C": 3, "NT": 4}  # endplay Denom order

FULL_CONTRACTS = [("PASS", None, None)] + [
    (f"{lvl}{st}", lvl, si)
    for lvl in range(1, 8)
    for si, st in enumerate(STRAINS)
]

# small starting subset suggested by the plan
SUBSET_NAMES = ["PASS", "2C", "2D", "2H", "2S", "2NT", "3C", "3D", "3H",
                "3S", "3NT", "4H", "4S", "5C", "5D", "6NT"]
SUBSET_CONTRACTS = [c for c in FULL_CONTRACTS if c[0] in SUBSET_NAMES]


def score_contract(level, strain_idx, tricks, vulnerable=False):
    """Duplicate score for declarer, undoubled. level=None means Pass -> 0."""
    if level is None:
        return 0
    needed = 6 + level
    if tricks >= needed:
        per = 20 if strain_idx <= 1 else 30
        trick_score = per * level + (10 if strain_idx == 4 else 0)
        total = trick_score
        total += 300 if not vulnerable else 500  # game bonus, fixed below
        if trick_score < 100:
            total = trick_score + 50              # partscore instead
        total += per * (tricks - needed)          # overtricks
        if level == 6:
            total += 500 if not vulnerable else 750
        elif level == 7:
            total += 1000 if not vulnerable else 1500
        return total
    down = needed - tricks
    return -(50 if not vulnerable else 100) * down


def build_score_table(contracts, vulnerable=False):
    """(n_contracts, 14) table: score for each contract given tricks 0..13."""
    tbl = np.zeros((len(contracts), 14), dtype=np.float32)
    for ci, (_, lvl, si) in enumerate(contracts):
        for t in range(14):
            tbl[ci, t] = score_contract(lvl, si, t, vulnerable)
    return tbl


def contract_strain_dd(contracts):
    """DD-table column index for each contract (Pass -> 0, score is 0 anyway)."""
    out = np.zeros(len(contracts), dtype=np.int64)
    for ci, (name, lvl, si) in enumerate(contracts):
        out[ci] = 0 if lvl is None else STRAIN_TO_DD[STRAINS[si]]
    return out
