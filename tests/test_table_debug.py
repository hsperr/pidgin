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


def test_a_lost_debug_game_reopens_from_the_address_bar(client):
    """After a restart (every deploy) the server no longer knows a /debug tab's game and
    deals it a plain board. The page must notice (`debug` false) and reopen the position
    in its address bar, hints as just clicked, not carry on as a / table with hands hidden."""
    d = client.post("/api/table/hints", json={"hints": True}, headers={"X-Game": "dbg-lost"}).json
    assert d["hints"] and not d["debug"] and d["debug_code"] is None
    assert any(s["hidden"] for s in d["seats"])
    page = client.get("/debug").data.decode()
    assert 'if(DEBUG && !d.debug){ boot({hints: d.hints ? "1" : "0"}); return; }' in page
    assert "debugBody(p, over)" in page


# ---- who plays what: by default the user holds all four chairs and the nets wait

def post(client, gid, url, **body):
    return client.post(url, json=body, headers={"X-Game": gid})


def test_you_hold_every_chair_by_default(client):
    d = load(client, "dbg-all", **HANDS, dealer="E", vul="ns", hints="1").json
    assert d["user_chairs"] == [0, 1, 2, 3] and d["debug_code"]["bots"] == ""
    assert d["view_seat"] == 2 and d["debug_code"]["seat"] == "S"
    # Every chair calls by hand, East first; the nets never move on their own, and the
    # hints follow the chair on turn, reading only that chair's cards.
    for want, call in zip([1, 2, 3, 0, 1, 2, 3], ["1H", "1NT", "P", "2C", "P", "2NT", "P"]):
        assert d["your_turn"] and d["to_play"] == want and d["user_seat"] == want
        assert d["hint"]["kind"] == "auction" and d["hint"]["seat"] == want
        assert d["your_hand"]["hcp"] == bidserver.tabledesk.hand_facts(
            bidserver.tabledesk.GAMES["dbg-all"]["bitmaps"][want])["hcp"]
        assert bidserver.tabledesk.known_seats(bidserver.tabledesk.GAMES["dbg-all"]) == {want}
        n = len(d["auction"])
        d = post(client, "dbg-all", "/api/table/call",
                 call=bidserver.tabledesk.CALL_WORDS[call]).json
        assert len(d["auction"]) == n + 1                        # nothing more than our call
    # Undo takes back the last call, whoever made it.
    d = post(client, "dbg-all", "/api/table/undo").json
    assert [a["name"] for a in d["auction"]][-1] == "2NT" and d["to_play"] == 3


def test_you_play_every_card_dummy_through_declarer(client):
    # 1NT by South, West leads: West, then North (dummy, played by South), East, South.
    d = load(client, "dbg-all2", **HANDS, dealer="S", auction="1NT-P-P-P", hints="1").json
    assert d["phase"] == "play" and d["to_play"] == 3 and d["user_seat"] == 3
    assert d["hint"]["kind"] == "play" and d["hint"]["seat"] == 3
    order = []
    for _ in range(4):
        order.append((d["to_play"], d["user_seat"]))
        assert d["your_turn"]
        d = post(client, "dbg-all2", "/api/table/card", card=d["legal_cards"].index(True)).json
    assert order == [(3, 3), (0, 2), (1, 1), (2, 2)]             # dummy's card is South's to pick
    # The step button: the net makes the one card on turn, and only that.
    n = d["trick_no"] * 4 + len(d["trick"])
    d = post(client, "dbg-all2", "/api/table/step").json
    assert d["trick_no"] * 4 + len(d["trick"]) == n + 1 and d["your_turn"]
    # Play to the end: the nets take every chair.
    d = post(client, "dbg-all2", "/api/table/finish").json
    assert d["phase"] == "over" and d["result"] is not None


def test_seat_and_bots_give_the_nets_chairs(client):
    d = load(client, "dbg-one", **HANDS, dealer="E", seat="S").json        # as on /
    assert d["user_chairs"] == [2] and d["debug_code"]["bots"] == "NEW"
    assert [a["name"] for a in d["auction"]] == []                          # stopped at the link
    d = post(client, "dbg-one", "/api/table/advance").json
    assert d["auction"] and d["your_turn"] and d["to_play"] == 2           # East bid by itself
    d = load(client, "dbg-mix", **HANDS, dealer="N", bots="EW", seat="N").json
    assert d["user_chairs"] == [0, 2] and d["view_seat"] == 0 and d["debug_code"]["bots"] == "EW"
    d = post(client, "dbg-mix", "/api/table/call", call=PASS).json         # North by hand
    assert [a["seat"] for a in d["auction"]] == [0, 1] and d["to_play"] == 2   # East by the net
    d = post(client, "dbg-mix", "/api/table/new_board").json
    assert d["user_chairs"] == [0, 2] and d["view_seat"] == 0               # new board keeps it
    assert load(client, "dbg-mix", bots="NESW").status_code == 400
    assert load(client, "dbg-mix", bots="EQ").status_code == 400
    d = load(client, "dbg-mix", bots="-", seat="E").json                    # all four, East at the bottom
    assert d["user_chairs"] == [0, 1, 2, 3] and d["view_seat"] == 1


def test_step_and_finish_on_the_main_table(client):
    d = client.post("/api/table/new_board", json={}, headers={"X-Game": "dbg-main"}).json
    assert d["user_chairs"] == [2] and d["view_seat"] == d["user_seat"] == 2
    assert post(client, "dbg-main", "/api/table/step").status_code == 400   # /debug only


# ---- the model menus: bm / pm in the readable link

NEW_BID_MODELS = ("hi3_s60k", "lo_s28k")


def test_new_bidding_models_are_listed_after_the_default(client):
    ids = list(engine.BID_MODELS)
    assert ids[0] == "D_cw_s75k" == engine.default_bid_model()          # / and the APIs
    assert ids[-2:] == list(NEW_BID_MODELS)
    d = load(client, "dbg-menu", **HANDS).json
    assert [m["id"] for m in d["models"]] == ids
    assert [m["id"] for m in d["play_models"]] == list(engine.PLAY_MODELS)


def test_model_link_round_trips(client):
    d = load(client, "dbg-bm", **HANDS, seat="S", model="lo_s28k", play_model="E48_leagueE").json
    assert (d["model"], d["play_model"]) == ("lo_s28k", "E48_leagueE")
    k = d["debug_code"]
    assert (k["bm"], k["pm"]) == ("lo_s28k", "E48_leagueE") and "m" not in k
    # What the page writes back opens the same models in a fresh tab.
    d = load(client, "dbg-bm2", **{x: k[x] for x in "nesw"}, seat=k["seat"],
             model=k["bm"], play_model=k["pm"]).json
    assert (d["model"], d["play_model"]) == ("lo_s28k", "E48_leagueE")
    # The defaults are left out of the link, and a link without them means the defaults,
    # whatever the tab had before.
    d = load(client, "dbg-bm2", **HANDS, seat="S").json
    assert (d["model"], d["play_model"]) == (engine.default_bid_model(), engine.default_play_model())
    assert (d["debug_code"]["bm"], d["debug_code"]["pm"]) == ("", "")
    # The menus (POST /api/table/models) change the link too; new board keeps the pick.
    d = post(client, "dbg-bm2", "/api/table/models", model="hi3_s60k", play_model="E48_leagueE").json
    assert (d["debug_code"]["bm"], d["debug_code"]["pm"]) == ("hi3_s60k", "E48_leagueE")
    d = post(client, "dbg-bm2", "/api/table/new_board").json
    assert (d["model"], d["debug_code"]["bm"]) == ("hi3_s60k", "hi3_s60k")
    page = client.get("/debug").data.decode()
    assert 'model: p.get("bm") || p.get("m") || ""' in page             # old links' m still works
    assert 'if(c.bm) q.push("bm="' in page


@pytest.mark.parametrize("body,words", [
    (dict(model="no-such-model"), "bidding model 'no-such-model' is not on this server"),
    (dict(play_model="no-such-model"), "card-play model 'no-such-model' is not on this server"),
])
def test_unknown_model_ids_are_refused(client, body, words):
    before = load(client, "dbg-unk", **HANDS, model="lo_s28k").json
    r = load(client, "dbg-unk", **HANDS, **body)
    assert r.status_code == 400 and words in r.json["error"]
    # The tab's game is untouched, and the menus ignore an unknown id.
    d = post(client, "dbg-unk", "/api/table/models", model="nope", play_model="nope").json
    assert (d["model"], d["play_model"]) == (before["model"], before["play_model"])


@pytest.mark.parametrize("bm", NEW_BID_MODELS)
def test_the_nets_bid_with_the_picked_model(client, monkeypatch, bm):
    used = []
    real = engine.choose_call
    monkeypatch.setattr(engine, "choose_call", lambda bot, *a, **k: used.append(bot.id) or real(bot, *a, **k))
    load(client, "dbg-use", **HANDS, dealer="E", vul="both", seat="S", model=bm)
    used.clear()
    post(client, "dbg-use", "/api/table/advance")                    # the bot chairs
    load(client, "dbg-use", **HANDS, dealer="W", model=bm)           # all four chairs yours
    post(client, "dbg-use", "/api/table/step")                       # Net: this call
    d = post(client, "dbg-use", "/api/table/finish").json            # Nets play to the end
    assert d["phase"] == "over" and used and set(used) == {bm}
    # The calls are the picked net's own greedy calls, from the real dealer and vulnerability.
    game = bidserver.tabledesk.GAMES["dbg-use"]
    bot = engine.BID_MODELS[bm]
    for i, call in enumerate(game["calls"]):
        seat = (game["dealer"] + i) % 4
        assert engine.choose_call(bot, game["bitmaps"][seat], game["calls"][:i],
                                  game["dealer"], game["vul"])[0] == call


def test_the_nets_play_with_the_picked_card_model(client, monkeypatch):
    used = []
    real = engine.choose_card
    monkeypatch.setattr(engine, "choose_card", lambda bot, *a, **k: used.append(bot.id) or real(bot, *a, **k))
    post(client, "dbg-card", "/api/table/hints", search=False)
    load(client, "dbg-card", **HANDS, dealer="S", auction="1NT-P-P-P", play_model="E48_leagueE")
    post(client, "dbg-card", "/api/table/step")
    post(client, "dbg-card", "/api/table/models", play_model="E48_wideleagueH")   # mid-hand switch
    d = post(client, "dbg-card", "/api/table/finish").json
    assert d["phase"] == "over"
    assert used[0] == "E48_leagueE" and set(used[1:]) == {"E48_wideleagueH"}


@pytest.mark.parametrize("bm", NEW_BID_MODELS)
def test_a_model_without_a_corpus_still_hints(client, bm):
    assert bm not in bidserver.CORPUS                 # no "what this call meant" table
    h = {"X-Game": "dbg-nocorp"}
    d = load(client, "dbg-nocorp", **HANDS, auction="1S-P", seat="S", hints="1", model=bm).json
    assert d["model"] == bm and d["hint"]["kind"] == "auction"
    assert d["hint"]["has_corpus"] is False and d["hint"]["corpus"] == []
    assert client.get("/api/table/state", headers=h).status_code == 200
    d = post(client, "dbg-nocorp", "/api/table/finish").json
    assert d["phase"] == "over"
    # The old bid desk API with the same model.
    g = {"X-Game": "dbg-nocorp-desk"}
    assert client.post("/api/model", json={"model": bm}, headers=g).status_code == 200
    assert client.get("/api/state", headers=g).status_code == 200


def test_the_main_table_is_unaffected(client):
    d = client.post("/api/table/new_board", json={}, headers={"X-Game": "dbg-plain"}).json
    assert not d["debug"] and d["debug_code"] is None
    assert d["model"] == "D_cw_s75k" and d["play_model"] == engine.default_play_model()
    assert d["code"]["m"] == "D_cw_s75k" and d["code"]["pm"] == d["play_model"]   # / link: m, pm
    page = client.get("/").data.decode()
    assert 'if(c.m) q.set("m", c.m);' in page
