"""emergent/bench.py: deals out, auctions in, a report page back.

    python3 -m pytest -q tests/test_bench.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bench, bidserver  # noqa: E402
from training.bridge.scoring import contract_score  # noqa: E402

HCP = {"A": 4, "K": 3, "Q": 2, "J": 1}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    bench.REPORT_DIR = str(tmp_path_factory.mktemp("bench"))
    bidserver.create_app()
    return bidserver.app.test_client()


def opener(hand: str) -> str:
    """A toy bot: 12+ HCP opens its longest suit at the 1 level, else Pass."""
    if sum(HCP.get(c, 0) for c in hand) < 12:
        return "P"
    suits = hand.split(".")
    longest = max(range(4), key=lambda s: (len(suits[s]), -s))
    return "1" + "SHDC"[longest]


def auction(board: dict, bot_side: str) -> str:
    """Only the bot's side ever bids: the first of its players with an opening opens, then all pass."""
    seats = "NESW"
    calls, start = [], seats.index(board["dealer"])
    for k in range(4):
        seat = seats[(start + k) % 4]
        call = opener(board["hands"][seat]) if seat in bot_side else "P"
        calls.append(call)
        if call != "P":
            return " ".join(calls + ["P", "P", "P"])
    return "P P P P"


def test_deals_are_fixed_and_hide_tricks(client):
    r = client.get("/apis/bench/deals?offset=0&limit=3").json
    assert r["total"] == bench.SET_SIZE and len(r["boards"]) == 3
    b = r["boards"][0]
    assert b["board"] == 1 and set(b["hands"]) == set("NESW") and "tricks" not in b
    cards = "".join(h.replace(".", "") for h in b["hands"].values())
    assert len(cards) == 52
    assert client.get("/apis/bench/deals?offset=0&limit=3").json == r
    pbn = client.get("/apis/bench/deals?format=pbn&limit=2").get_data(as_text=True)
    assert pbn.count("[Deal ") == 2 and '[Board "2"]' in pbn


def test_report_scores_double_dummy_and_renders(client):
    boards = client.get("/apis/bench/deals?limit=300").json["boards"]
    body = {"bot": "toy opener", "opponent": "pass bot", "public": True,
            "boards": [{"board": b["board"], "ns": auction(b, "NS"), "ew": auction(b, "EW")}
                       for b in boards]}
    r = client.post("/apis/bench/report", json=body)
    assert r.status_code == 200, r.json
    rid = r.json["id"]
    # the same match gives the same id
    assert client.post("/apis/bench/report", json=body).json["id"] == rid

    report = client.get(f"/bench/report/{rid}?format=json").json
    assert report["weak"]["boards"] == 300
    assert report["openings"]["A"]["openings"] > 0 and report["openings"]["B"]["openings"] == 0
    # board 1 by hand: table 1 NS score minus table 2 NS score, from the DD tricks
    d = bench.deals()
    z = bench.score_match(body)["z"]
    for t in (1, 2):
        c, decl = int(z[f"c{t}"][0]), int(z[f"d{t}"][0])
        if c < 0:
            assert z[f"ns{t}"][0] == 0
            continue
        vul = bool(d["vul"][0] in ((1, 3) if decl % 2 == 0 else (2, 3)))
        tricks = int(d["tricks"][0, decl, (3, 2, 1, 0, 4)[c % 5]])
        want = contract_score(c // 5 + 1, c % 5, tricks, 0, vul) * (1 if decl % 2 == 0 else -1)
        assert z[f"ns{t}"][0] == want
    assert report["weak"]["imps_per_board"] == pytest.approx(
        float(np.mean(z["imps_A"])))

    page = client.get(f"/bench/report/{rid}").get_data(as_text=True)
    assert "<title>toy opener vs pass bot</title>" in page and '"weak"' in page
    assert "toy opener" in client.get("/bench").get_data(as_text=True)   # public: listed


def test_bad_matches_are_refused(client):
    boards = [{"board": i, "ns": "P P P P", "ew": "P P P P"} for i in range(1, 201)]
    ok = client.post("/apis/bench/report", json={"boards": boards})
    assert ok.status_code == 200 and ok.json["imps_per_board"] == 0
    for bad, word in ((boards[:10], "200"),
                      (boards[:-1] + [{"board": 1, "ns": "P P P P", "ew": "P P P P"}], "twice"),
                      (boards[:-1] + [{"board": 200, "ns": "1H 1C P P P", "ew": "P P P P"}], "illegal"),
                      (boards[:-1] + [{"board": 200, "ns": "1H P P", "ew": "P P P P"}], "not ended")):
        r = client.post("/apis/bench/report", json={"boards": bad})
        assert r.status_code == 400 and word in r.json["error"], r.json
    assert client.get("/bench/report/0123456789abcdef").status_code == 404


def test_spec_matches_the_routes(client):
    spec = client.get("/apis/bench/openapi.json").json
    assert set(spec["paths"]) == {"/apis/bench/deals", "/apis/bench/report", "/bench/report/{id}"}
    guide = client.get("/apis/bench/agent.md").get_data(as_text=True)
    assert "/apis/bench/report" in guide and "{" not in guide.split("## 1.")[0]


def test_a_bot_name_cannot_break_out_of_the_page(client):
    boards = [{"board": i, "ns": "P P P P", "ew": "P P P P"} for i in range(1, 201)]
    rid = client.post("/apis/bench/report", json={"bot": "</script><b>x", "boards": boards}).json["id"]
    page = client.get(f"/bench/report/{rid}").get_data(as_text=True)
    assert "</script><b>" not in page
