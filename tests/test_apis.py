"""Conformance tests for emergent/apis.py: Brill's checklist (seat-api.html §9) and robot.php.

    python3 -m pytest -q tests/test_apis.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emergent import apis, bidserver  # noqa: E402

SEATS, SUITS, RANKS = "NESW", "SHDC", "AKQJT98765432"


@pytest.fixture(scope="module")
def client():
    bidserver.create_app()
    return bidserver.app.test_client()


def hand_text(cards):
    return ".".join("".join(RANKS[c % 13] for c in sorted(cards) if c // 13 == s) for s in range(4))


def deal(seed):
    owners = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), 13))
    return [[c for c in range(52) if owners[c] == s] for s in range(4)]


def play_board(client, hands, dealer, vul, board="1"):
    """Drive one board the way Brill does: one stateless request per decision."""
    ctx, calls = "", []
    for _ in range(60):
        seat = (dealer + len(calls)) % 4
        r = client.get("/apis/brill/bid", query_string={
            "seat": SEATS[seat], "dealer": SEATS[dealer], "vul": vul, "ctx": ctx,
            "hand": hand_text(hands[seat]), "board": board, "matchtype": "IMP"})
        assert r.status_code == 200, r.json
        tok = r.json["bid"]
        assert len(tok) == 2
        assert "explanation" in r.json
        ctx += tok
        calls.append(apis.parse_call(tok))
        if apis.AuctionState.from_calls(calls).ended:
            break
    contract = apis.playdesk.contract_from_calls(calls, dealer)
    if contract is None:
        return ctx, None
    declarer = contract["declarer"]
    dummy = (declarer + 2) % 4
    trump = contract["trump"]
    leader = (declarer + 1) % 4
    r = client.get("/apis/brill/lead", query_string={
        "seat": SEATS[leader], "dealer": SEATS[dealer], "vul": vul, "ctx": ctx,
        "hand": hand_text(hands[leader]), "board": board})
    assert r.status_code == 200, r.json
    played = [apis.parse_card(r.json["card"])]
    assert played[0] in hands[leader]
    left = {s: set(hands[s]) for s in range(4)}
    left[leader].discard(played[0])
    while len(played) < 52:
        _, turn = apis.trick_seats(played, leader, trump)
        asked = declarer if turn == dummy else turn           # dummy convention
        r = client.get("/apis/brill/play", query_string={
            "seat": SEATS[asked], "dealer": SEATS[dealer], "vul": vul, "ctx": ctx,
            "hand": hand_text(hands[asked]), "dummy": hand_text(hands[dummy]),
            "played": "".join(apis.card_name(c) for c in played), "board": board})
        assert r.status_code == 200, r.json
        card = apis.parse_card(r.json["card"])
        assert card in left[turn], f"{r.json['card']} not held by {SEATS[turn]}"
        pos = len(played) % 4
        if pos:
            led = played[len(played) - pos] // 13
            if any(c // 13 == led for c in left[turn]):
                assert card // 13 == led, "revoke"
        left[turn].discard(card)
        played.append(card)
    return ctx, played


def test_root(client):
    assert client.get("/apis/brill/").status_code == 200
    assert client.get("/apis/brill").status_code == 200


@pytest.mark.parametrize("seed,dealer,vul", [(1, 0, "None"), (2, 1, "NS"), (3, 2, "EW"), (4, 3, "All")])
def test_full_board(client, seed, dealer, vul):
    ctx, played = play_board(client, deal(seed), dealer, vul, board=str(seed))
    assert ctx.endswith("------")
    if played is not None:
        assert sorted(played) == list(range(52))


def test_opening_and_matchtype(client):
    hands = deal(7)
    for mt in ("IMP", "MP", None):
        q = {"seat": "N", "dealer": "N", "vul": "None", "ctx": "", "hand": hand_text(hands[0])}
        if mt:
            q["matchtype"] = mt
        r = client.get("/apis/brill/bid", query_string=q)
        assert r.status_code == 200 and len(r.json["bid"]) == 2


def test_meanings_accepted(client):
    hands = deal(8)
    r = client.get("/apis/brill/bid", query_string={
        "seat": "E", "dealer": "N", "vul": "None", "ctx": "1S", "hand": hand_text(hands[1]),
        "meanings": '["5+ spades, 12-21 HCP"]', "partnerMeanings": "true"})
    assert r.status_code == 200


def test_same_request_same_answer(client):
    hands = deal(9)
    q = {"seat": "S", "dealer": "N", "vul": "NS", "ctx": "1D--", "hand": hand_text(hands[2])}
    assert client.get("/apis/brill/bid", query_string=q).json == client.get("/apis/brill/bid", query_string=q).json


def test_bid_matches_desk(client):
    """Dealer North, nobody vulnerable is the desk's setting: the API must pick the same call."""
    desk = bidserver
    rng = np.random.default_rng(0)
    for _ in range(20):
        owners = rng.permutation(np.repeat(np.arange(4), 13))
        game = desk.new_game(owners)
        while not desk.auction(game).ended:
            seat = len(game["calls"]) % 4
            ctx = "".join(apis.bid_token(c, "brill") for c in game["calls"])
            hand = [c for c in range(52) if owners[c] == seat]
            r = client.get("/apis/brill/bid", query_string={
                "seat": SEATS[seat], "dealer": "N", "vul": "None", "ctx": ctx, "hand": hand_text(hand)})
            api_call = apis.parse_call(r.json["bid"])
            assert api_call == desk.net_pick(game)
            desk.add_call(game, api_call)


def test_filler_never_changes_the_card(client):
    """The unseen cards are random filler: the chosen card and its probabilities must not move."""
    _, bot = apis.play_model(None)
    hands = deal(11)
    calls = [apis.parse_call(t) for t in ("1S", "--", "2S", "--", "4S", "--", "--", "--")]
    ctx_play = []
    # play 9 cards with seed A, then compare the 10th decision under several fillers
    declarer = apis.playdesk.contract_from_calls(calls, 0)["declarer"]
    dummy = (declarer + 2) % 4
    leader = (declarer + 1) % 4
    for i in range(9):
        _, turn = apis.trick_seats(ctx_play, leader, 0)
        asked = declarer if turn == dummy else turn
        card, _ = apis.choose_card(bot, seat=asked, hand=hands[asked],
                                   dummy=hands[dummy] if ctx_play else None, played=ctx_play,
                                   calls=calls, dealer=0, vul=(False, False), seed_text="x",
                                   search=False)
        ctx_play.append(card)
    _, turn = apis.trick_seats(ctx_play, leader, 0)
    asked = declarer if turn == dummy else turn
    answers = {apis.choose_card(bot, seat=asked, hand=hands[asked], dummy=hands[dummy],
                                played=ctx_play, calls=calls, dealer=0, vul=(False, False),
                                seed_text=f"filler{k}", search=False)[1][0]
               for k in range(6)}
    assert len({round(p, 6) for _, p in answers}) == 1 and len({c for c, _ in answers}) == 1


def test_declarer_card_comes_from_search(client):
    """Declarer's cards go through the PIMC search, as on /table, and stay legal."""
    _, bot = apis.play_model(None)
    hands = deal(11)
    calls = [apis.parse_call(t) for t in ("1S", "--", "2S", "--", "4S", "--", "--", "--")]
    declarer = apis.playdesk.contract_from_calls(calls, 0)["declarer"]
    dummy = (declarer + 2) % 4
    played = [apis.choose_card(bot, seat=(declarer + 1) % 4, hand=hands[(declarer + 1) % 4],
                               dummy=None, played=[], calls=calls, dealer=0,
                               vul=(False, False), seed_text="x")[0]]
    before = bot.searcher().solves
    card, _ = apis.choose_card(bot, seat=declarer, hand=hands[declarer], dummy=hands[dummy],
                               played=played, calls=calls, dealer=0, vul=(False, False),
                               seed_text="x")
    assert bot.searcher().solves > before
    assert card in hands[dummy]


def test_errors_are_400_json(client):
    r = client.get("/apis/brill/bid", query_string={"seat": "N", "dealer": "N", "vul": "None", "ctx": ""})
    assert r.status_code == 400 and "hand" in r.json["error"]
    hands = deal(12)
    r = client.get("/apis/brill/bid", query_string={
        "seat": "E", "dealer": "N", "vul": "None", "ctx": "", "hand": hand_text(hands[1])})
    assert r.status_code == 400
    r = client.get("/apis/brill/play", query_string={
        "seat": "N", "dealer": "N", "vul": "None", "ctx": "--------", "hand": hand_text(hands[0]),
        "played": "SA"})
    assert r.status_code == 400


def test_bbo_php(client):
    hands = deal(13)
    q = {"botstyle": "advanced", "pov": "S", "d": "N", "v": "-", "h": "1c-p",
         **{k: hand_text(hands[i]).lower() for i, k in enumerate("nesw")}}
    r = client.get("/apis/bbo.php", query_string=q)
    assert r.status_code == 200 and r.mimetype == "text/xml"
    m = re.search(r'type="bid"\s+bid="([^"]*)"', r.get_data(as_text=True))
    assert m and apis.parse_call(m.group(1)) >= 0
    q["pov"] = "E"
    assert client.get("/apis/bbo.php", query_string=q).status_code == 400


def test_bbo_php_play_example(client):
    """The robot.php request from BBO: E declares 4H, S led H9, dummy (W) is on turn."""
    q = {"botstyle": "simplistic", "pov": "E", "d": "N", "v": "-",
         "s": "j6.952.k8765.qj6", "w": "q4.jt6.q432.kt75", "n": "k8532.k7.aj9.942",
         "e": "at97.aq843.t.a83",
         "h": "1s-1n-p-2c-p-2h-p-2n-p-3c-p-3h-p-4h-p-p-p-H9"}
    r = client.get("/apis/bbo.php", query_string=q)
    assert r.status_code == 200, r.get_data(as_text=True)
    m = re.search(r'type="play"\s+card="([^"]*)"', r.get_data(as_text=True))
    assert m and m.group(1) in ("HJ", "HT", "H6")        # dummy must follow hearts


def test_bbo_php_full_board(client):
    """Bid and play a whole board through robot.php-style requests only."""
    hands = deal(31)
    dealer = 2
    q = {k: hand_text(hands[i]).lower() for i, k in enumerate("nesw")}
    q.update(d=SEATS[dealer], v="b")
    h = []
    while True:
        calls = [apis.parse_call(t) for t in h]
        if calls and apis.AuctionState.from_calls(calls).ended:
            break
        r = client.get("/apis/bbo.php", query_string={**q, "pov": SEATS[(dealer + len(h)) % 4], "h": "-".join(h)})
        h.append(re.search(r'bid="([^"]*)"', r.get_data(as_text=True)).group(1).lower())
    contract = apis.playdesk.contract_from_calls(calls, dealer)
    if contract is None:
        return
    declarer = contract["declarer"]
    dummy = (declarer + 2) % 4
    trump = contract["trump"]
    played, left = [], {s: set(hands[s]) for s in range(4)}
    while len(played) < 52:
        _, turn = apis.trick_seats(played, (declarer + 1) % 4, trump)
        pov = declarer if turn == dummy else turn
        r = client.get("/apis/bbo.php", query_string={**q, "pov": SEATS[pov],
                                                      "h": "-".join(h + [apis.card_name(c) for c in played])})
        assert r.status_code == 200, r.get_data(as_text=True)
        card = apis.parse_card(re.search(r'card="([^"]*)"', r.get_data(as_text=True)).group(1))
        assert card in left[turn]
        left[turn].discard(card)
        played.append(card)
    assert sorted(played) == list(range(52))
