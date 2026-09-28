"""One bot engine behind every surface: the bid desk (/), /table, /play and the APIs.

    python3 -m pytest -q tests/test_engine.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import apis, bidserver, engine, playdesk  # noqa: E402
from emergent.deck import RANKS, decode_cards, encode_deal  # noqa: E402

SEATS = "NESW"


@pytest.fixture(scope="module")
def client():
    if not engine.BID_MODELS:
        bidserver.create_app()
    return bidserver.app.test_client()


def owners_of(seed):
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(4), 13))


def hand_text(owners, seat):
    return ".".join("".join(RANKS[c % 13] for c in range(s * 13, s * 13 + 13) if owners[c] == seat)
                    for s in range(4))


def table_board(client, gid, owners, seat, model, auction="", search=False):
    """Play a board on /table from `seat`, the user always taking the hint's move."""
    h = {"X-Game": gid}
    client.post("/api/table/hints", json={"hints": True, "search": search}, headers=h)
    s = client.post("/api/table/load", json={"deal": encode_deal(owners), "seat": SEATS[seat],
                                             "model": model, "auction": auction, "hints": 1},
                    headers=h).json
    assert "error" not in s, s
    s = client.post("/api/table/advance", headers=h).json
    for _ in range(200):
        if s["phase"] == "over":
            break
        if s["your_turn"]:
            sg = s["suggest"]
            url = "/api/table/call" if sg["kind"] == "call" else "/api/table/card"
            s = client.post(url, json={sg["kind"]: sg["action"]}, headers=h).json
        else:
            s = client.post("/api/table/advance", headers=h).json
    return [bidserver.CALL_CHARS.find(ch) for ch in s["code"]["a"]], decode_cards(s["code"]["p"])


@pytest.mark.parametrize("model", ["D_cw_s75k", "E21_last"])
def test_same_calls_everywhere(client, model):
    """Dealer North, nobody vulnerable: bid desk, /table and the Brill API bid alike."""
    if model not in engine.BID_MODELS:
        pytest.skip(f"{model} is not loaded")
    for seed in range(4):
        owners = owners_of(seed)
        h = {"X-Game": f"desk-{model}-{seed}"}
        client.post("/api/load", json={"deal": encode_deal(owners), "model": model}, headers=h)
        desk = [a["call"] for a in client.post("/api/auto_end", headers=h).json["auction"]]

        table, _ = table_board(client, f"table-{model}-{seed}", owners, seed % 4, model)
        assert table == desk

        ctx, api = "", []
        while not (api and apis.AuctionState.from_calls(api).ended):
            seat = len(api) % 4
            r = client.get("/apis/brill/bid", query_string={
                "seat": SEATS[seat], "dealer": "N", "vul": "None", "ctx": ctx,
                "hand": hand_text(owners, seat), "model": model})
            ctx += r.json["bid"]
            api.append(apis.parse_call(r.json["bid"]))
        assert api == desk


def test_same_cards_everywhere(client):
    """Search off: /table, /play and the API play the same cards on the same board."""
    owners = owners_of(21)
    calls = [apis.parse_call(t) for t in ("1S", "--", "2S", "--", "4S", "--", "--", "--")]
    auction = "".join(bidserver.CALL_CHARS[c] for c in calls)
    contract = playdesk.contract_from_calls(calls, 0)
    declarer, dummy = contract["declarer"], (contract["declarer"] + 2) % 4
    # the user sits dummy on /table, so the nets play every card
    _, table = table_board(client, "cards-table", owners, dummy, None, auction)
    assert len(table) == 52

    h = {"X-Game": "cards-play"}
    client.post("/api/play/setup", json={"deal": encode_deal(owners), "auction": auction,
                                         "dealer": 0, "vul": "none"}, headers=h)
    play = decode_cards(client.post("/api/play/auto", headers=h).json["code"]["p"])
    assert play == table

    _, bot = apis.play_model(None)
    hands = [[c for c in range(52) if owners[c] == s] for s in range(4)]
    for i in range(52):
        _, turn = apis.trick_seats(table[:i], (declarer + 1) % 4, contract["trump"])
        asked = declarer if turn == dummy else turn
        card, _ = apis.api_card(bot, seat=asked, hand=hands[asked],
                                dummy=hands[dummy] if i else None, played=table[:i], calls=calls,
                                dealer=0, vul=(False, False), seed_text="x", search=False)
        assert card == table[i], f"card {i + 1}"


def test_search_is_on_by_default_on_table_and_apis(client):
    """CONFIG.search is the default of /table and of the APIs; both reach the searcher."""
    assert engine.CONFIG.search and engine.CONFIG.samples == 20
    assert engine.CONFIG.budget_ms == 900 and engine.CONFIG.defence == "all"
    assert engine.CONFIG.defence_from == 2
    owners = owners_of(22)
    calls = [apis.parse_call(t) for t in ("1S", "--", "2S", "--", "4S", "--", "--", "--")]
    contract = playdesk.contract_from_calls(calls, 0)
    declarer, dummy = contract["declarer"], (contract["declarer"] + 2) % 4
    leader = (declarer + 1) % 4
    _, bot = apis.play_model(None)
    player = engine.searcher(bot)

    # /table: the user leads, then declarer (a net) plays dummy's card and searches
    h = {"X-Game": "search-table"}
    s = client.post("/api/table/load", json={
        "deal": encode_deal(owners), "seat": SEATS[leader],
        "auction": "".join(bidserver.CALL_CHARS[c] for c in calls)}, headers=h).json
    assert s["search"] is True
    lead = next(c for c in range(52) if s["legal_cards"][c])
    before = player.solves
    s = client.post("/api/table/card", json={"card": lead}, headers=h).json
    assert "error" not in s and player.solves > before

    # the Brill API: declarer asked for dummy's card after the lead
    hands = [[c for c in range(52) if owners[c] == seat] for seat in range(4)]
    before = player.solves
    r = client.get("/apis/brill/play", query_string={
        "seat": SEATS[declarer], "dealer": "N", "vul": "None", "ctx": "1S--2S--4S------",
        "hand": hand_text(owners, declarer), "dummy": hand_text(owners, dummy),
        "played": apis.card_name(lead)})
    assert r.status_code == 200, r.json
    assert player.solves > before and apis.parse_card(r.json["card"]) in hands[dummy]


def test_play_desk_shows_the_plain_net(client):
    """/play is the net viewer: its own games do not search."""
    h = {"X-Game": "plain-play"}
    client.post("/api/play/setup", json={"bench": 0}, headers=h)
    before = engine.searcher(apis.play_model(None)[1]).solves
    client.post("/api/play/net_card", headers=h)
    assert engine.searcher(apis.play_model(None)[1]).solves == before
