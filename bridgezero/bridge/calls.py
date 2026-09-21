"""Canonical call vocabulary shared by rules, model, and checkpoints."""

STRAINS = ("C", "D", "H", "S", "NT")
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

