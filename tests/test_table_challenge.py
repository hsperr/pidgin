"""The /table challenge: four boards in South against a table of four nets.

    python3 -m pytest -q tests/test_table_challenge.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bidserver  # noqa: E402

PASS = 35


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def finish_board(c, h):
    """The user passes and plays the first legal card; the nets do the rest."""
    for _ in range(200):
        d = c.get("/api/table/state", headers=h).json
        if d["phase"] == "over":
            return d
        if not d["your_turn"]:
            c.post("/api/table/finish", json={}, headers=h)
        elif d["phase"] == "auction":
            c.post("/api/table/call", json={"call": PASS}, headers=h)
        else:
            c.post("/api/table/card", json={"card": d["legal_cards"].index(True)}, headers=h)
    raise AssertionError("board did not finish")


def test_challenge_and_its_link(client):
    h = {"X-Game": "chal-1"}
    d = client.post("/api/table/challenge", json={"boards": 2}, headers=h).json
    ch = d["challenge"]
    assert ch["n"] == 2 and d["user_seat"] == 2 and not d["hints"]
    assert all(b["bot"] is None for b in ch["boards"])       # nothing to peek at yet
    assert client.post("/api/table/hints", json={"hints": True}, headers=h).status_code == 400
    assert client.post("/api/table/restart", json={}, headers=h).status_code == 400
    assert client.post("/api/table/challenge/next", json={}, headers=h).status_code == 400

    d = finish_board(client, h)
    row = d["challenge"]["boards"][0]
    assert row["bot"] is not None
    assert row["imps"] == bidserver.tabledesk.imps(row["mine"]["ns_score"] - row["bot"]["ns_score"])
    assert d["challenge"]["boards"][1]["bot"] is None
    d = client.post("/api/table/challenge/next", json={}, headers=h).json
    d = finish_board(client, h)
    assert d["challenge"]["done"] and d["challenge"]["last"]
    assert d["challenge"]["total"] == sum(b["imps"] for b in d["challenge"]["boards"])

    # A friend opens the link: same deals, the same other table, replayed not re-run.
    k = d["challenge"]["code"]
    body = {"deals": k["c"], "auctions": k["ca"], "played": k["cp"],
            "model": k["m"], "play_model": k["pm"], "search": k["r"]}
    f = client.post("/api/table/challenge/load", json=body, headers={"X-Game": "chal-2"})
    assert f.status_code == 200 and f.json["challenge"]["code"] == k

    bad = dict(body, played=k["cp"][1:])
    r = client.post("/api/table/challenge/load", json=bad, headers={"X-Game": "chal-3"})
    assert r.status_code == 400

    # New board leaves the challenge.
    d = client.post("/api/table/new_board", json={}, headers=h).json
    assert d["challenge"] is None
