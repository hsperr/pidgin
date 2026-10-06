"""Canonical call vocabulary shared by rules, model, and checkpoints."""

STRAINS = ("C", "D", "H", "S", "NT")

# Bids count up C,D,H,S,NT; cards, trick tables and the solver count down S,H,D,C,NT.
# Everything that scores a contract has to cross between the two, and the permutation is
# its own inverse -- index 3 <-> 0, 2 <-> 1 -- so a swapped call is invisible to a reader
# and silent at runtime. It would score a 4S contract against the club trick count: no
# crash, a plausible number, several hundred points wrong, and every reward, ceiling and
# IMP in the project shifted with it. Derived rather than written out, and imported
# rather than copied, so there is one place to be wrong.
CARD_SUITS = ("S", "H", "D", "C", "NT")
STRAIN_PERM = tuple(CARD_SUITS.index(s) for s in STRAINS)
CONTRACTS = tuple(
    (f"{level}{strain}", level, strain_idx)
    for level in range(1, 8)
    for strain_idx, strain in enumerate(STRAINS)
)
PASS = 35
DOUBLE = 36
REDOUBLE = 37
N_ACTIONS = 38
ACTION_NAMES = tuple(x[0] for x in CONTRACTS) + ("P", "X", "XX")
NAME_TO_ACTION = {name: i for i, name in enumerate(ACTION_NAMES)}
NAME_TO_ACTION.update({"PASS": PASS, "DBL": DOUBLE, "DOUBLE": DOUBLE,
                       "RDBL": REDOUBLE, "REDOUBLE": REDOUBLE})


def parse_call(text: str) -> int:
    key = text.strip().upper().replace("N", "NT") if text.strip().upper().endswith("N") else text.strip().upper()
    if key not in NAME_TO_ACTION:
        raise ValueError(f"unknown bridge call {text!r}")
    return NAME_TO_ACTION[key]


def format_call(action: int) -> str:
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action out of range: {action}")
    return ACTION_NAMES[action]

