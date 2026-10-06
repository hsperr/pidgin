"""Shared helpers for the scripts: load a bidder the way the server does.

A bidder is a team id from server/models/teams.json (PidginV1, PidginV2, BRL) or a
checkpoint file. Bots come from server/emergent, so a script bids exactly like the site.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
MODELS = SERVER / "models"
sys.path.insert(0, str(SERVER))
os.environ.setdefault("PLAY_SEARCH", "0")      # scripts bid; no card-play search at import

from emergent import bidserver, engine  # noqa: E402
from emergent.deck import NAMES, RANKS, SUITS  # noqa: E402
from training.bridge.calls import DOUBLE, PASS, REDOUBLE  # noqa: E402

SEATS = "NESW"


def teams() -> dict:
    path = MODELS / "teams.json"
    return {m["id"]: m for m in json.load(open(path))} if path.exists() else {}


def load_bidder(name: str):
    """A team id or a checkpoint path -> the server's bidding bot."""
    t = teams().get(name)
    if t:
        path, family = MODELS / t["bid"], t.get("bid_family")
    else:
        path, family = Path(name), "brl" if name.endswith(".npz") else None
        if not path.exists():
            path = MODELS / name
    if not path.exists():
        known = ", ".join(teams()) or "none: run scripts/get_models.sh"
        sys.exit(f"no model {name!r}. Teams: {known}. Or pass a checkpoint file.")
    bot = bidserver.BrlBot(str(path)) if family == "brl" else bidserver.FourSeatBot(str(path))
    bot.id = name
    return bot


def call_name(c: int) -> str:
    return "P" if c == PASS else "X" if c == DOUBLE else "XX" if c == REDOUBLE else NAMES[c]


def parse_call(tok: str) -> int:
    t = tok.upper().replace("PASS", "P").replace("N", "NT").replace("NTT", "NT")
    if t == "P":
        return PASS
    if t == "X":
        return DOUBLE
    if t == "XX":
        return REDOUBLE
    if t in NAMES:
        return NAMES.index(t)
    raise ValueError(f"unknown call {tok!r}; use 1C..7NT, P, X, XX")


def parse_hand(text: str):
    """'AKQ2.JT9.876.543' (spades.hearts.diamonds.clubs) -> 52 card bits."""
    import numpy as np
    parts = text.upper().replace("10", "T").split(".")
    if len(parts) != 4:
        raise ValueError("hand needs four suits: spades.hearts.diamonds.clubs")
    bits = np.zeros(52, dtype=np.float32)
    for s, cards in enumerate(parts):
        for r in cards.replace("-", ""):
            bits[s * 13 + RANKS.index(r)] = 1
    if bits.sum() != 13:
        raise ValueError(f"hand has {int(bits.sum())} cards, not 13")
    return bits


def parse_vul(text: str) -> tuple[bool, bool]:
    t = text.lower()
    return {"none": (False, False), "ns": (True, False), "ew": (False, True),
            "both": (True, True), "all": (True, True)}[t]


__all__ = ["ROOT", "MODELS", "SEATS", "SUITS", "engine", "load_bidder", "call_name",
           "parse_call", "parse_hand", "parse_vul", "teams"]
