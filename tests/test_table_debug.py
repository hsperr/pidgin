"""/debug: the table with all four hands up, a readable link, a real dealer and vulnerability.

    python3 -m pytest -q tests/test_table_debug.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import bidserver, engine  # noqa: E402

PASS = 35
HANDS = {"n": "AQJ32.Q8.T32.K83", "e": "T54.KT63.875.AJ9",
         "s": "K987.AJ.AKQJ.QT4", "w": "6.97542.964.7652"}


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def load(client, gid, **body):
    return client.post("/api/table/debug", json=body, headers={"X-Game": gid})


def test_page_is_the_table(client):
    r = client.get("/debug")
    assert r.status_code == 200
    assert r.data == client.get("/").data          # one page, no second copy of the logic


def test_readable_link_round_trips(client):
    d = load(client, "dbg-1", **HANDS, dealer="E", vul="ns", auction="1S-P-2C-P",
             play="", seat="S").json
    assert d["debug"] and d["dealer"] == 1 and d["vul"] == "ns" and d["phase"] == "auction"
    assert [a["name"] for a in d["auction"]] == ["1S", "Pass", "2C", "Pass"]
    assert [a["seat"] for a in d["auction"]] == [1, 2, 3, 0]          # East deals
    assert all(not s["hidden"] for s in d["seats"])                   # all four face up
    k = d["debug_code"]
    assert {x: k[x] for x in "nesw"} == HANDS
    assert (k["dealer"], k["vul"], k["seat"], k["auction"]) == ("E", "ns", "S", ["1S", "P", "2C", "P"])
    assert not d["hints"] and d["hint"] is None and d["me"] is None
    assert d["code"]["dr"] == 1 and d["code"]["v"] == "ns"             # the compact link carries them too


def test_play_and_a_finished_board(client):
    # 1NT by South (dealer), all pass; West leads.
    d = load(client, "dbg-2", **HANDS, dealer="S", vul="both", auction="1NT-P-P-P",
             play="h7-hq-hk-ha", seat="S").json
    assert d["contract"]["label"] == "1NT" and d["contract"]["declarer"] == 2
    assert d["debug_code"]["play"] == ["H7", "HQ", "HK", "HA"]
    assert d["debug_code"]["auction"] == ["1NT", "P", "P", "P"]
    h = {"X-Game": "dbg-2"}
    for _ in range(200):
        if d["phase"] == "over":
            break
        if not d["your_turn"]:
            d = client.post("/api/table/finish", json={}, headers=h).json
        else:
            d = client.post("/api/table/card", json={"card": d["legal_cards"].index(True)}, headers=h).json
    assert d["phase"] == "over"
    r = d["result"]
    raw = bidserver.tabledesk.contract_score(1, 4, r["tricks"], 0, True)   # vulnerable
    assert r["score"] == raw


def test_missing_hands_are_dealt(client):
    d = load(client, "dbg-3", n=HANDS["n"]).json
    k = d["debug_code"]
    assert k["n"] == HANDS["n"] and k["dealer"] == "N" and k["vul"] == "none" and k["seat"] == "S"
    held = "".join(k[x].replace(".", "").replace("-", "") for x in "nesw")
    assert len(held) == 52
    d = load(client, "dbg-3").json                       # nothing at all: a random deal
    assert d["debug"] and len(set(d["debug_code"][x] for x in "nesw")) == 4


@pytest.mark.parametrize("body, words", [
    (dict(HANDS, n="AQJ32.Q8.T32.K8"), "North has 12 cards"),
    (dict(HANDS, n="AQJ32.Q8.T32"), "four suits"),
    (dict(HANDS, n="AQJ32.Q8.T32.K8Z"), "not a rank"),
    (dict(HANDS, e="A54.KT63.875.AJ9"), "in both North's and East's"),
    (dict(HANDS, auction="1S-P-1H"), "not legal"),
    (dict(HANDS, auction="1S-P-9Z"), "not a call"),
    (dict(HANDS, auction="P-P-P-P", play="SA"), "passed-out"),
    (dict(HANDS, auction="1S-P-P-P", play="SA"), "not legal"),     # South is not on lead
    (dict(HANDS, dealer="Q"), "dealer"),
    (dict(HANDS, vul="red"), "vul"),
    (dict(HANDS, m="no-such-model"), "not on this server"),
])
def test_bad_links_say_why(client, body, words):
    body = {("model" if k == "m" else k): v for k, v in body.items()}
    r = load(client, "dbg-bad", **body)
    assert r.status_code == 400 and words in r.json["error"]


def test_debug_never_rates(client):
    h = {"X-Game": "dbg-4"}
    load(client, "dbg-4", **HANDS)
    for url in ("/api/table/challenge", "/api/table/challenge/job",
                "/api/table/challenge/start", "/api/table/challenge/load"):
        r = client.post(url, json={"boards": 1}, headers=h)
        assert r.status_code == 400 and "debug" in r.json["error"], url
    # New board stays a debug table, with its dealer and vulnerability.
    load(client, "dbg-4", dealer="W", vul="ew")
    d = client.post("/api/table/new_board", json={}, headers=h).json
    assert d["debug"] and d["dealer"] == 3 and d["vul"] == "ew"


def test_the_nets_see_dealer_and_vul(client, monkeypatch):
    seen = []
    real = engine.choose_call
    monkeypatch.setattr(engine, "choose_call",
                        lambda bot, hand, calls, dealer=0, vul=(False, False):
                        seen.append((dealer, vul)) or real(bot, hand, calls, dealer, vul))
    load(client, "dbg-5", **HANDS, dealer="E", vul="ew", seat="S")
    seen.clear()                       # the tab's first default board bid on its own
    client.post("/api/table/advance", json={}, headers={"X-Game": "dbg-5"})
    assert seen and all(x == (1, (False, True)) for x in seen)


def test_table_links_carry_dealer_and_vul(client):
    d = client.post("/api/table/new_board", json={}, headers={"X-Game": "dbg-6"}).json
    assert d["dealer"] == 0 and d["vul"] == "none" and not d["debug"]    # / is unchanged
    assert any(s["hidden"] for s in d["seats"])
    body = {"deal": d["code"]["d"], "seat": "S", "dealer": "2", "vul": "all"}
    d = client.post("/api/table/load", json=body, headers={"X-Game": "dbg-6"}).json
    assert d["dealer"] == 2 and d["vul"] == "all" and d["code"]["dr"] == 2 and not d["debug"]
    r = client.post("/api/table/load", json=dict(body, dealer="7"), headers={"X-Game": "dbg-6"})
    assert r.status_code == 400
    d = client.post("/api/table/new_board", json={}, headers={"X-Game": "dbg-6"}).json
    assert d["dealer"] == 0 and d["vul"] == "none"                        # a new board on / goes back


def test_hints_on_the_debug_table(client):
    h = {"X-Game": "dbg-7"}
    d = load(client, "dbg-7", **HANDS, dealer="E", vul="ns", auction="1S", seat="S").json
    assert not d["hints"] and d["debug_code"]["hints"] == 0 and d["your_turn"]
    d = client.post("/api/table/hints", json={"hints": True}, headers=h).json
    assert d["hints"] and d["debug_code"]["hints"] == 1 and d["hint"]["kind"] == "auction"
    assert "measured with nobody vulnerable" in d["hint"]["conditions_note"]
    assert d["me"] is None and d["challenge"] is None
    # The hints read only South's chair, though the page shows all four hands.
    game = bidserver.tabledesk.GAMES["dbg-7"]
    assert bidserver.tabledesk.known_seats(game) == {2}
    assert bidserver.tabledesk.visible_seats(game) == {0, 1, 2, 3}
    # The readable link carries the toggle; nobody vulnerable needs no note.
    d = load(client, "dbg-8", **HANDS, auction="1S-P", seat="S", hints="1").json
    assert d["hints"] and d["hint"] is not None and d["hint"]["conditions_note"] is None
    assert load(client, "dbg-8", **HANDS, hints="2").status_code == 400


def test_rollout_turns_the_table_for_the_dealer(client):
    """`explain` deals from North; with East dealing its seats must come back as the real ones."""
    import time
    h = {"X-Game": "dbg-9"}
    load(client, "dbg-9", **HANDS, dealer="E", vul="ew", auction="1S", seat="S", hints="1")
    r = client.post("/api/table/explain", json={"samples": 32}, headers=h).json
    for _ in range(300):
        if r.get("status") in ("done", "error"):
            break
        time.sleep(0.1)
        r = client.get("/api/table/explain", query_string={"job": r["job"]}).json
    assert r["status"] == "done", r
    assert sorted(p["seat"] for p in r["result"]["picture"]) == [0, 1, 3]     # everyone but South
