"""Table: play a whole board against the nets, with teaching hints.

Routes live on the bid desk's Flask app (`register(app)` from bidserver). The table
is the site's only page; the bid desk and play desk keep their APIs, not their pages.

    GET  /  and  /table         the page
    GET  /api/table/state       everything the page draws
    POST /api/table/new_board   deal again (optionally from a different chair)
    POST /api/table/load        open a shared link: that deal, chair, models, calls, cards
    GET  /debug                 the same page in debug mode: all hands up, readable link
    POST /api/table/debug       open a debug link (see "/debug" below)
    POST /api/table/restart     same cards, auction from the top
    POST /api/table/call        the user's call
    POST /api/table/card        the user's card
    POST /api/table/advance     let the nets act until it is the user's turn again
    POST /api/table/finish      let them run the rest of the deal out
    POST /api/table/step        /debug: the nets make the one call or card on turn
    POST /api/table/undo        take back the user's last call or card
    POST /api/table/claim       claim every trick left; the solver checks and plays it out
    POST /api/table/hints       teaching on/off, solver peek on/off
    POST /api/table/models      pick the bidding and card-play checkpoints
    POST /api/table/explain     queue the "where does it go" rollout (newest wins)
    GET  /api/table/explain     collect it
    POST /api/table/challenge/job       deal a challenge in the background (the page uses this)
    GET  /api/table/challenge/progress  how far that is
    POST /api/table/challenge/cancel    stop it
    POST /api/table/challenge/start     sit down at a dealt challenge

Player routes (/api/me, /api/leaderboard) and the challenge rating are in `players`.

The user holds one chair; their partner and both opponents are nets. The play
follows a real table: a declaring user plays both their hand and dummy's, and a
user who is dummy plays nothing while partner, the net, plays both hands.
The bidding net comes from `models/models.json`, the E48 card-play net from
`models/play_models.json`. Dealer is North and nobody is vulnerable, exactly as
on the other two desks, because the offline "what this call means" corpus was
built under those conditions. A link (`dr`, `v`) or the debug page can set both;
they reach the nets as real inputs.

Nothing here reimplements a net. Every call and card the nets make comes from
`emergent.engine` (the play on a `playdesk`-shaped game dict), so a board played
here is played by the same code the other desks and the APIs use.

What the hints are allowed to say is the whole point of this file. Three sources,
kept apart in the payload and on the page:

- `net`       - the deployed net's own output for the seat on turn. It is given
                exactly the user's information set, so this is advice, not a peek.
- `corpus`    - `models/corpus_<model id>.json`, a million greedy self-play
                auctions summarised by prefix. What a call *held*, measured.
- `position`  - facts read off the cards the user can already see. Arithmetic,
                not opinion.

A fourth, `solver`, is double dummy and does look at all four hands. It is off by
default and always labelled. The net has no explanation head, so nothing in here
ever claims a reason the net did not have; the "about that card" notes describe
the card, not a motive.

One request may carry several of the nets' moves — a whole trick, or the rest of
the deal when the user is dummy. The page walks to them one at a time on its own;
nothing here paces anything, and `advance` stopping at the end of a trick is only
there to keep each response small.
"""
import os
import secrets
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from flask import has_request_context, jsonify, request, send_from_directory

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import DOUBLE, PASS, REDOUBLE
from bridgezero.bridge.scoring import contract_score, dd_par_score, imps
from emergent import engine, players, playdesk
from emergent.deck import (HCP_W, N_CALLS, NAMES, RANKS, SEAT_NAMES, STRAINS, SUITS,
                           TRUMP_TO_BID_STRAIN, beats, call_name, call_token, card_name, trick_best)
from emergent.deck import deal_owners, owners_to_bitmaps
from emergent.teaching import situations as teaching

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "bidserver_static")

SUIT_WORDS = {"S": "spade", "H": "heart", "D": "diamond", "C": "club"}
DEALER = 0                                 # North, like both other desks
VUL_NAMES = {(False, False): "nobody vulnerable", (True, False): "North-South vulnerable",
             (False, True): "East-West vulnerable", (True, True): "both sides vulnerable"}
MAX_GAMES = 300

GAMES = OrderedDict()
LOCK = threading.Lock()
EXPLAIN = {"lock": threading.Lock(), "jobs": OrderedDict(), "next": None, "worker": None}
MAX_EXPLAIN_JOBS = 40


def rank_text(card):
    return "10" if card % 13 == 4 else RANKS[card % 13]


def card_text(card):
    """Plain-words card, for a sentence: 'the ♠K'."""
    return f"{'♠♥♦♣'[card // 13]}{rank_text(card)}"


# ------------------------------------------------------------------ the board

DESK = None      # the bid desk module itself, handed over by `load`


def bid_models():
    return engine.BID_MODELS


def play_models():
    return engine.PLAY_MODELS


def new_board(user_seat=2, model=None, play_model=None, hints=False, peek=False,
              owners=None, board_no=1, dealer=DEALER, vul=(False, False), debug=False, bots=None):
    """A fresh game dict. North deals and nobody is vulnerable unless a link or the
    debug page says otherwise; the nets read both as real inputs either way.

    `bots` (debug only) is the set of chairs the nets play; the user acts for every
    other chair, and `user_seat` then follows the turn (see `follow_turn`). None, as on
    /, means the nets play every chair but `user_seat`."""
    bm, pm = bid_models(), play_models()
    if owners is None:
        owners = deal_owners(np.random.default_rng())
    owners = np.asarray(owners, dtype=np.int64)
    return {
        "owners": owners,
        "bitmaps": owners_to_bitmaps(owners),
        "user_seat": int(user_seat) % 4,
        "view_seat": int(user_seat) % 4,     # the chair at the bottom of the page
        "bots": None if bots is None else frozenset(int(b) % 4 for b in bots),
        "calls": [],
        "model": model if model in bm else engine.default_bid_model(),
        "play_model": play_model if play_model in pm else engine.default_play_model(),
        "dealer": int(dealer) % 4,
        "vul": (bool(vul[0]), bool(vul[1])),
        # /debug: all four hands face up, never a challenge or a rating.
        "debug": bool(debug),
        "hints": bool(hints),
        "peek": bool(peek),
        "search": engine.CONFIG.search,     # the page's toggle; see engine.choose_card

        "board_no": int(board_no),
        "pg": None,          # the playdesk game dict, built when the auction ends
        "tricks": None,      # (4, 5) double dummy table, computed with `pg`
    }


def reset_board(game):
    """Same cards, auction from the top."""
    game["calls"] = []
    game["pg"] = None
    game["tricks"] = None
    game.pop("_batch", None)
    game.pop("claim", None)


def auction_of(game):
    return AuctionState.from_calls(game["calls"], dealer=game["dealer"])


def bid_bot(game):
    return bid_models().get(game["model"])


def legal_calls(game):
    """38 bools: the rules, narrowed to the calls this bidding model can make."""
    st = auction_of(game)
    bot = bid_bot(game)
    return st.legal_mask()[:N_CALLS].tolist() if bot is None else engine.legal_calls(bot, st)


def net_call(game):
    """The bidding bot's call for the seat on turn, or None without a model.

    On /debug the search toggle covers the auction too: with it on, the call comes from
    the bidding search (emergent/bidsearch.py). Everywhere else the net's own call."""
    bot = bid_bot(game)
    if bot is None:
        return None
    seat = auction_of(game).turn
    args = (bot, game["bitmaps"][seat], game["calls"], game["dealer"], game["vul"])
    if game["debug"] and engine.search_on(game.get("search")):
        return engine.search_call(*args)
    return engine.choose_call(*args)[0]


def bid_view(game):
    """The bidding net's whole output for the seat on turn, on that seat's own cards."""
    bot = bid_bot(game)
    if bot is None:
        return None, legal_calls(game)
    legal = legal_calls(game)
    return bot.view({"calls": game["calls"], "bitmaps": game["bitmaps"],
                     "dealer": game["dealer"], "vul": game["vul"]}, legal), legal


def add_call(game, call):
    if auction_of(game).ended or not 0 <= call < N_CALLS or not legal_calls(game)[call]:
        return False
    game["calls"].append(int(call))
    if auction_of(game).ended:
        begin_play(game)
    return True


def begin_play(game):
    """Hand the finished auction to the card-play desk's own game dict."""
    st = auction_of(game)
    if st.passed_out or st.last_contract < 0:
        return
    game["pg"] = playdesk.new_game(owners=game["owners"], calls=list(game["calls"]),
                                   dealer=game["dealer"], vul=game["vul"],
                                   model=game["play_model"])
    game["pg"]["search"] = game.get("search", engine.CONFIG.search)
    game["tricks"] = game["pg"]["tricks"]


def batch_for(game):
    """The `PlayBatch` for the cards played so far, rebuilt only when they change.

    `playdesk.batch_of` replays the whole deal, and one request asks for the
    position a dozen times, so cache it against the number of cards played.
    """
    pg = game["pg"]
    if pg is None:
        return None
    cached = game.get("_batch")
    if cached is not None and cached[0] is pg and cached[1] == len(pg["played"]):
        return cached[2]
    batch = playdesk.batch_of(pg)
    game["_batch"] = (pg, len(pg["played"]), batch)
    return batch


def phase(game):
    st = auction_of(game)
    if not st.ended:
        return "auction"
    if st.passed_out or game["pg"] is None:
        return "over"
    return "over" if batch_for(game).done else "play"


# --------------------------------------------------------------- who plays what

def declarer_dummy(game):
    pg = game["pg"]
    return pg["declarer"], (pg["declarer"] + 2) % 4


def user_chairs(game):
    """The chairs the user acts for: `user_seat` on /, on /debug every chair the nets do not play."""
    if game.get("bots") is None:
        return {game["user_seat"]}
    return {0, 1, 2, 3} - game["bots"]


def player_of(game, seat):
    """The chair that chooses the card for `seat`: declarer for dummy, else the seat itself."""
    if game["pg"] is not None and seat == declarer_dummy(game)[1]:
        return declarer_dummy(game)[0]
    return seat


def user_plays(game, seat):
    """Whether the user chooses the card for ``seat``, by the rules of a real table.

    Declarer plays both of the declaring side's hands, so a declaring user plays
    dummy's cards too, and a user who is dummy plays none: partner, the net,
    plays both hands while dummy lies face up. A defender plays their own hand.
    On /debug the user may hold several chairs (`user_chairs`); the same rule holds.
    """
    return player_of(game, seat) in user_chairs(game)


def follow_turn(game):
    """/debug with more than one chair: the user is whichever of them acts now, so the
    hints, the "you" on the page and what may be seen follow the turn. Declarer acts
    for dummy. At the end of the board it is the bottom chair again."""
    chairs = user_chairs(game)
    if len(chairs) < 2:
        return
    ph = phase(game)
    if ph == "auction":
        actor = auction_of(game).turn
    elif ph == "play":
        actor = player_of(game, int(batch_for(game).to_play()[0]))
    else:
        actor = game["view_seat"]
    if actor in chairs or ph == "over":
        game["user_seat"] = actor


def user_on_turn(game):
    ph = phase(game)
    if ph == "auction":
        return auction_of(game).turn in user_chairs(game)
    if ph == "play":
        return user_plays(game, int(batch_for(game).to_play()[0]))
    return False


def user_role(game):
    """'declarer' / 'dummy' / 'defender' once there is a contract."""
    if game["pg"] is None:
        return None
    declarer, dummy = declarer_dummy(game)
    me = game["user_seat"]
    return "declarer" if me == declarer else "dummy" if me == dummy else "defender"


def advance(game, limit=60):
    """Let the nets act until the user is on turn. Stops when a trick fills up.

    The page animates whatever comes back one card at a time, so a request may
    safely carry several. Stopping at the end of a trick keeps that batch small
    and bounded, which matters when the user is dummy and the nets would
    otherwise play all thirteen tricks in a single request.
    """
    acted = 0
    for _ in range(limit):
        if phase(game) not in ("auction", "play") or user_on_turn(game):
            return acted
        if not net_act(game):
            return acted
        acted += 1
        if phase(game) == "play" and len(game["pg"]["played"]) % 4 == 0 and game["pg"]["played"]:
            return acted                          # a trick just filled up
    return acted


def net_act(game):
    """One call or card by the nets for the seat on turn, whoever holds it. False if none."""
    ph = phase(game)
    if ph == "auction":
        call = net_call(game)
        return call is not None and add_call(game, call)
    if ph != "play":
        return False
    pg, batch = game["pg"], batch_for(game)
    card = playdesk.net_card(pg, playdesk.contracts_of(pg), batch)
    if card is None or not playdesk.play_card(pg, card, batch):
        return False
    game.pop("_batch", None)                      # `play_card` moved the batch on
    return True


# ------------------------------------------------------------ undo and claim

def played_seats(pg):
    """The seat that played each card so far, in order."""
    batch = playdesk.batch_of({**pg, "played": []})
    seats = []
    for card in pg["played"]:
        seats.append(int(batch.to_play()[0]))
        batch.play(torch.tensor([card]))
    return seats


def can_undo(game):
    """Something of the user's to take back. Every card the user played comes after a
    call of theirs, so one call of theirs in the auction is enough to know."""
    chairs = user_chairs(game)
    return not game.get("challenge") and any((game["dealer"] + i) % 4 in chairs for i in range(len(game["calls"])))


def undo(game):
    """Wind back to just before the user's last call or card, with the user on turn.

    Everything the nets did after it goes too; they choose again when play goes on.
    A claimed board goes back to the moment of the claim. A user who is dummy has
    no cards of their own, so for them it takes back the last call.
    """
    pg = game["pg"]
    if pg is not None and pg["played"]:
        claim = game.pop("claim", None)
        if claim is not None:
            cut = claim["at"]
        else:
            mine = [i for i, seat in enumerate(played_seats(pg)) if user_plays(game, seat)]
            cut = mine[-1] if mine else None
        if cut is not None:
            del pg["played"][cut:]
            game.pop("_batch", None)
            return True
    chairs = user_chairs(game)
    mine = [i for i in range(len(game["calls"])) if (game["dealer"] + i) % 4 in chairs]
    if not mine:
        return False
    reset_calls = game["calls"][:mine[-1]]
    reset_board(game)
    game["calls"] = reset_calls
    return True


def can_claim(game):
    """Not in a challenge: a refused claim tells the user the defence still has a trick."""
    return (playdesk.SOLVER is None and not game.get("challenge")
            and phase(game) == "play" and user_on_turn(game))


def claim(game):
    """Claim every trick still to come, the one on the table included.

    Double dummy decides, the way a director would: the claim stands only if the
    user's side takes all of them against any defence. Then the solver plays the
    rest out, both sides at their best, so the board ends like any other.
    """
    pg = game["pg"]
    left = 13 - batch_for(game).trick_no
    per_card = playdesk.solve_here(playdesk.solver_deal(pg))
    if not per_card or max(per_card.values()) < left:
        return False
    game["claim"] = {"at": len(pg["played"]), "trick": 14 - left, "tricks": left}
    while not batch_for(game).done:
        per_card = playdesk.solve_here(playdesk.solver_deal(pg))
        best = max(per_card.values())
        playdesk.play_card(pg, min(c for c, v in per_card.items() if v == best), batch_for(game))
        game.pop("_batch", None)
    return True


# ------------------------------------------------------------------- the view

def visible_seats(game):
    """The chairs whose cards go into the payload: what the user knows, and on /debug all four."""
    return {0, 1, 2, 3} if game["debug"] else known_seats(game)


def known_seats(game):
    """The chairs the user may see at a real table. The hints read only these, on /debug too."""
    ph = phase(game)
    if ph == "over":
        return {0, 1, 2, 3}
    me = game["user_seat"]
    if ph == "auction":
        return {me}
    _, dummy = declarer_dummy(game)
    # Your own hand, and dummy once the opening lead is on the table. That is what
    # every chair sees at a real table, declarer's included.
    return {me} | ({dummy} if len(game["pg"]["played"]) > 0 else set())


def seats_view(game, seen):
    """Hands for the page. A hidden chair gives up nothing but how many cards it holds."""
    batch = batch_for(game)
    unplayed = None if batch is None else batch.unplayed[0].numpy()
    out = []
    for seat in range(4):
        left = 13 if unplayed is None else int(
            sum(1 for c in range(52) if game["owners"][c] == seat and unplayed[c]))
        if seat in seen:
            info = playdesk.hand_info(game["owners"], seat, unplayed)
        else:
            info = {"seat": seat, "name": SEAT_NAMES[seat], "hcp": None, "shape": None,
                    "suits": None, "spent": None, "left": left}
        info["hidden"] = seat not in seen
        info["left"] = left
        out.append(info)
    return out


def hand_facts(bitmap52):
    sh = np.asarray(bitmap52).reshape(4, 13)
    shape = [int(x) for x in sh.sum(1)]
    srt = sorted(shape, reverse=True)
    balanced = srt[0] <= 4 or (srt[0] == 5 and srt[1] == 3)
    balanced = bool(balanced and srt[3] >= 2)
    return {"hcp": int((sh * HCP_W).sum()), "shape": shape, "balanced": balanced,
            "longest": SUITS[int(np.argmax(shape))]}


def contract_view(game):
    pg = game["pg"]
    if pg is None:
        return None
    declarer, dummy = declarer_dummy(game)
    strain = STRAINS[pg["trump"]]
    return {
        "level": pg["level"], "strain": strain, "trump": pg["trump"],
        "doubled": pg["doubled"], "label": f"{pg['level']}{strain}" + ["", "X", "XX"][pg["doubled"]],
        "declarer": declarer, "declarer_name": SEAT_NAMES[declarer],
        "dummy": dummy, "dummy_name": SEAT_NAMES[dummy],
        "needs": pg["level"] + 6,
    }


def trick_views(game):
    """Every finished trick, the one on the table, and the last one that was taken."""
    batch = batch_for(game)
    if batch is None:
        return [], [], None, None
    declarer = game["pg"]["declarer"]
    done_tricks = []
    for t in range(batch.trick_no):
        leader = int(batch.trick_winner[0, t - 1]) if t else (declarer + 1) % 4
        cards = batch.history[0, t * 4:t * 4 + 4].tolist()
        done_tricks.append({
            "no": t, "leader": leader, "winner": int(batch.trick_winner[0, t]),
            "cards": [{**playdesk.card_parts(c), "seat": (leader + k) % 4}
                      for k, c in enumerate(cards)],
        })
    leader = int(batch.leader[0])
    current = [{**playdesk.card_parts(int(c)), "seat": (leader + k) % 4}
               for k, c in enumerate(batch.trick_cards()[0].tolist())]
    return done_tricks, current, leader, (done_tricks[-1] if done_tricks else None)


def result_view(game):
    """Duplicate score for the board, and how it sits against the double dummy par."""
    st = auction_of(game)
    if not st.ended:
        return None
    me = game["user_seat"]
    my_side = me % 2
    if game["tricks"] is None:                       # a passed-out board never built one
        game["tricks"] = playdesk.dd_table(game["owners"])
    tricks = game["tricks"]
    par = int(dd_par_score(tricks, *game["vul"])) if tricks is not None else None
    if st.passed_out or game["pg"] is None:
        my_par = None if par is None else (par if my_side == 0 else -par)
        return {"passed_out": True, "ns_score": 0, "my_score": 0,
                "par": my_par, "par_ns": par, "par_mine": my_par,
                "par_delta": None if my_par is None else 0 - my_par,
                "imps": None if par is None else imps(0 - my_par),
                "made": None, "tricks": None, "needed": None, "dd_tricks": None}
    pg = game["pg"]
    batch = batch_for(game)
    if not batch.done:
        return None
    made = int(batch.declarer_tricks()[0])
    declarer = pg["declarer"]
    raw = contract_score(pg["level"], TRUMP_TO_BID_STRAIN[pg["trump"]], made, pg["doubled"],
                         game["vul"][declarer % 2])
    ns = raw if declarer % 2 == 0 else -raw
    mine = ns if my_side == 0 else -ns
    par_mine = None if par is None else (par if my_side == 0 else -par)
    dd_tricks = None if tricks is None else int(tricks[declarer][pg["trump"]])
    return {
        "passed_out": False, "tricks": made, "needed": pg["level"] + 6,
        "delta": made - (pg["level"] + 6), "made": made >= pg["level"] + 6,
        "score": raw, "ns_score": ns, "my_score": mine,
        "declaring_side": "NS" if declarer % 2 == 0 else "EW",
        "my_side": "NS" if my_side == 0 else "EW",
        "we_declared": declarer % 2 == my_side,
        # `par` is the headline number: double dummy par from the user's side, the
        # one the end-of-board panel compares against. `par_ns` and `par_mine` are
        # kept because they were here first.
        "par": par_mine, "par_ns": par, "par_mine": par_mine,
        "par_delta": None if par_mine is None else mine - par_mine,
        "imps": None if par_mine is None else imps(mine - par_mine),
        "dd_tricks": dd_tricks,
    }


def state_dump(game):
    follow_turn(game)
    ph = phase(game)
    st = auction_of(game)
    seen = visible_seats(game)
    pg = game["pg"]
    me = game["user_seat"]
    batch = batch_for(game)
    turn = None
    if ph == "auction":
        turn = st.turn
    elif ph == "play":
        turn = int(batch.to_play()[0])
    done_tricks, current, leader, last = trick_views(game)
    bm, pmd = bid_models(), play_models()
    out = {
        "phase": ph,
        "board_no": game["board_no"],
        "user_seat": me, "user_seat_name": SEAT_NAMES[me],
        "view_seat": game["view_seat"], "user_chairs": sorted(user_chairs(game)),
        "partner_seat": (me + 2) % 4, "partner_name": SEAT_NAMES[(me + 2) % 4],
        "role": user_role(game),
        "to_play": turn,
        "your_turn": user_on_turn(game),
        "seats": seats_view(game, seen),
        "your_hand": hand_facts(game["bitmaps"][me]),
        "dealer": game["dealer"], "dealer_name": SEAT_NAMES[game["dealer"]],
        "vul": playdesk.vul_code(*game["vul"]), "vul_name": VUL_NAMES[game["vul"]],
        "auction": [{"seat": (game["dealer"] + i) % 4, "call": c, "name": call_name(c)}
                    for i, c in enumerate(game["calls"])],
        "auction_over": st.ended, "passed_out": bool(st.ended and st.passed_out),
        "contract": contract_view(game),
        "legal_calls": legal_calls(game) if ph == "auction" else None,
        "legal_cards": None,
        "tricks": done_tricks, "trick": current, "trick_leader": leader,
        "last_trick": last,
        "trick_no": 0 if batch is None else batch.trick_no,
        "tricks_won": None,
        "hints": game["hints"], "peek": game["peek"], "search": game.get("search", engine.CONFIG.search),
        "model": game["model"], "play_model": game["play_model"],
        "models": [{"id": b.id, "label": b.label} for b in bm.values()],
        "play_models": [{"id": b.id, "label": b.label} for b in pmd.values()],
        "meta": ("bidding: " + (bm[game["model"]].label if game["model"] in bm else "none")
                 + " · cards: "
                 + (pmd[game["play_model"]].label if game["play_model"] in pmd else "none")),
        "meta_long": ((bm[game["model"]].info if game["model"] in bm else "no bidding model")
                      + " · "
                      + (pmd[game["play_model"]].info if game["play_model"] in pmd else "no play model")),
        "hint": None, "suggest": None,
        "code": link_code(game),
        "debug": game["debug"],
        "debug_code": debug_code(game) if game["debug"] else None,
        "result": result_view(game),
        "review": None,
        "challenge": challenge_view(game),
        "can_undo": can_undo(game),
        "can_claim": False,
        "claim": game.get("claim"),
        "me": players.me(players.current_id()) if has_request_context() and not game["debug"] else None,
    }
    if pg is not None:
        declarer = pg["declarer"]
        out["tricks_won"] = {
            "ns": int(batch.tricks_won[0, 0]), "ew": int(batch.tricks_won[0, 1]),
            "declaring": int(batch.tricks_won[0, declarer % 2]),
            "defending": int(batch.tricks_won[0, 1 - declarer % 2]),
            "mine": int(batch.tricks_won[0, me % 2]),
        }
        if ph == "play":
            out["legal_cards"] = batch.legal()[0].tolist()
            out["can_claim"] = can_claim(game)
    if ph == "over" and pg is not None and game["hints"]:
        out["review"] = user_review(game)
    # With hints off the payload carries no suggestion of any kind: the page cannot
    # show what it was never sent.
    if out["your_turn"] and game["hints"]:
        out["hint"] = auction_hint(game) if ph == "auction" else play_hint(game)
        out["suggest"] = suggest_from_hint(out["hint"])
    return out


# ----------------------------------------------------------- auction teaching

def corpus_table(game):
    return DESK.CORPUS.get(game["model"]) if DESK is not None else None


def corpus_at(game, calls):
    """The offline self-play table for one exact prefix, or None."""
    table = corpus_table(game)
    if table is None:
        return None
    return table.get("-".join(call_token(c) for c in calls))


def modal_call(entry):
    """The call self-play made most often at a position, as (token, stats)."""
    if not entry:
        return None
    return max(entry["calls"].items(), key=lambda kv: kv[1]["share"])


def partner_reply(game, candidate):
    """What partner says next if the user makes `candidate`, from the corpus.

    The corpus is keyed by the whole call sequence, so partner's node is three
    calls further on. We walk the single most likely opponent call to get there
    and hand the page that assumption to print, rather than hiding it.
    """
    calls = list(game["calls"]) + [candidate]
    lho = corpus_at(game, calls)
    pick = modal_call(lho)
    if pick is None:
        return None
    lho_call, lho_stats = pick
    nxt = calls + [token_to_call(lho_call)]
    partner = corpus_at(game, nxt)
    if partner is None:
        return None
    return {"assumed": {"call": lho_call, "share": lho_stats["share"], "n": lho_stats["n"]},
            "position": "-".join(call_token(c) for c in nxt),
            "n": partner["n"], "calls": partner["calls"]}


def token_to_call(token):
    if token == "P":
        return PASS
    if token == "X":
        return DOUBLE
    if token == "XX":
        return REDOUBLE
    return NAMES.index(token)


def partner_last(game):
    """What partner's most recent call promised, straight out of the corpus."""
    me = game["user_seat"]
    partner = (me + 2) % 4
    for i in range(len(game["calls"]) - 1, -1, -1):
        if (game["dealer"] + i) % 4 != partner:
            continue
        entry = corpus_at(game, game["calls"][:i])
        token = call_token(game["calls"][i])
        if entry is None or token not in entry["calls"]:
            return {"call": call_name(game["calls"][i]), "stats": None,
                    "position": "-".join(call_token(c) for c in game["calls"][:i]) or "(opening)",
                    "n": None}
        return {"call": call_name(game["calls"][i]), "stats": entry["calls"][token],
                "position": "-".join(call_token(c) for c in game["calls"][:i]) or "(opening)",
                "n": entry["n"]}
    return None


def fits_hand(stats, facts):
    """Does the user's hand sit inside what this call showed? Our rule, spelled out.

    Points inside the 5th-95th percentile band, and - for a call that named a
    suit - at least four cards in it when self-play held four or more most of the
    time. Nothing subtler: the corpus only stores a band and a few averages.
    """
    if not stats:
        return None
    lo, _, hi = stats["hcp"]
    ok = lo <= facts["hcp"] <= hi
    why = [("points", ok, f"{facts['hcp']} points vs {lo:g}–{hi:g}")]
    promised = promised_length(stats)
    if promised:
        want, share = promised
        held = facts["shape"][SUITS.index(stats["suit"])]
        suit_ok = held >= want
        ok = ok and suit_ok
        why.append((f"{stats['suit']} length", suit_ok,
                    f"{held} card{'' if held == 1 else 's'}, {int(round(share * 100))}% "
                    f"of these hands held {want}+"))
    return {"fits": bool(ok), "checks": [{"what": w, "ok": bool(o), "detail": d} for w, o, d in why]}


# --------------------------------------------- saying the corpus out loud
#
# Templates only. Every number in these sentences is read out of a corpus entry,
# never invented: a confident wrong sentence teaches a confident wrong habit. The
# judgement is in which clause gets used and how it is worded, and the thresholds
# that drive those choices are named constants below so they can be argued with.
# Anything the corpus does not measure - stoppers, controls, what a call asks for
# rather than shows - is simply not said.

SUIT_WORDS_LONG = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}
COUNT_WORD = {4: "four", 5: "five", 6: "six"}
PROMISE_AT = 0.50        # a length counts as promised once this many hands hold it
BALANCED_HIGH = 0.75     # above this share of balanced hands we say "balanced"
BALANCED_LOW = 0.15      # below it we say "unbalanced"; in between we say nothing
RUNNER_UP_AT = 0.10      # the net's second choice is named once it is this likely


def call_glyph(name):
    """A call as it should read inside a sentence."""
    if name in ("Pass", "P"):
        return "Pass"
    if name == "X":
        return "Double"
    if name == "XX":
        return "Redouble"
    if name.endswith("NT"):
        return name
    return name[0] + {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}[name[-1]]


def points_words(stats):
    """'8 to 11 points', straight off the 5th and 95th percentiles."""
    lo, _, hi = stats["hcp"]
    return f"{int(round(lo))} to {int(round(hi))} points"


def promised_length(stats):
    """The longest holding most of these hands actually had: (cards, share), or None.

    `corpus.py` stores the share of hands holding at least 4, at least 5 and at
    least 6 cards of the named suit, in that order. Read it as anything else and
    the sentence promises a suit the bid never showed.
    """
    if not stats.get("suit") or not stats.get("suit_len"):
        return None
    p4, p5, p6 = stats["suit_len"]
    for n, share in ((6, p6), (5, p5), (4, p4)):
        if share >= PROMISE_AT:
            return n, share
    return None


def length_words(stats):
    """'five or more hearts', or an honest caveat when no length is promised."""
    if not stats.get("suit") or not stats.get("suit_len"):
        return None
    suit = SUIT_WORDS_LONG[stats["suit"]]
    promised = promised_length(stats)
    if promised:
        return {"promise": True, "text": f"{COUNT_WORD[promised[0]]} or more {suit}"}
    share = stats["suit_len"][0]
    return {"promise": False,
            "text": f"four or more {suit} only about {int(round(share * 100))}% of the time"}


def cards_words(n, suit_letter):
    """'1 heart', '3 hearts' — so the sentence never says 'only 1 hearts'."""
    word = SUIT_WORDS_LONG[suit_letter]
    return f"{n} {word[:-1] if n == 1 else word}"


def shape_words(stats):
    """'a balanced hand', when the measured share is lopsided enough to be worth saying."""
    b = stats.get("bal")
    if b is None:
        return None
    if b >= BALANCED_HIGH:
        return "a balanced hand"
    if b <= BALANCED_LOW:
        return "an unbalanced hand"
    return None


def holding_words(stats):
    """The measured picture of one call: '8 to 11 points and five or more hearts'."""
    out = points_words(stats)
    length, shape = length_words(stats), shape_words(stats)
    if length and length["promise"]:
        out += " and " + length["text"]          # a named suit beats a shape adjective
    elif shape:
        out += " and " + shape
    if length and not length["promise"]:
        out += f", and {length['text']}"
    return out


def showed_sentence(who, name, stats, opened):
    """'North opened 1♠. Hands like that usually have 12 to 17 points and five or more spades.'"""
    if name == "Pass":
        head = f"{who} passed"
    elif name == "X":
        head = f"{who} doubled"
    elif name == "XX":
        head = f"{who} redoubled"
    else:
        head = f"{who} {'opened' if opened else 'bid'} {call_glyph(name)}"
    return f"{head}. Hands like that usually have {holding_words(stats)}."


def would_show_sentence(name, stats):
    """The same picture, addressed to the player about to make the call."""
    what = holding_words(stats)
    if name == "Pass":
        return f"Passing here shows {what}."
    if name == "X":
        return f"Doubling here shows {what}."
    if name == "XX":
        return f"Redoubling here shows {what}."
    return f"If you bid {call_glyph(name)} you would be showing {what}."


def fit_sentence(stats, facts):
    """How the user's own hand sits against that band. The comparison is arithmetic."""
    lo, _, hi = stats["hcp"]
    n = facts["hcp"]
    if n < lo:
        return f"You hold {n}, a little light for that."
    if n > hi:
        return f"You hold {n}, more than that usually shows."
    promised = promised_length(stats)          # test the length the sentence promised
    if promised:
        held = facts["shape"][SUITS.index(stats["suit"])]
        if held < promised[0]:
            return (f"Your {n} points fit, but you hold only "
                    f"{cards_words(held, stats['suit'])}.")
    return f"Your {n} points fit that."


def recent_calls(game, limit=2):
    """Partner's last call and the one on the user's right, said in words.

    Those two are what should change the user's mind. A call the corpus never saw
    gets no sentence at all: there is nothing measured to report.
    """
    me = game["user_seat"]
    wanted = [((me + 2) % 4, "Your partner"), ((me + 3) % 4, "The player on your right")]
    out = []
    for seat, who in wanted:
        for i in range(len(game["calls"]) - 1, -1, -1):
            if (game["dealer"] + i) % 4 != seat:
                continue
            before = game["calls"][:i]
            entry = corpus_at(game, before)
            token = call_token(game["calls"][i])
            stats = (entry or {}).get("calls", {}).get(token)
            opened = game["calls"][i] < PASS and not any(c < PASS for c in before)
            if stats:
                out.append({
                    "seat": seat, "seat_name": SEAT_NAMES[seat],
                    "call": call_name(game["calls"][i]),
                    "position": "-".join(call_token(c) for c in before) or "(opening)",
                    "stats": stats, "n": entry["n"],
                    "sentence": showed_sentence(who, call_name(game["calls"][i]), stats, opened),
                })
            break
    return out[:limit]


def percent_words(p):
    """'78%', or '<1%' so a live option never reads as zero."""
    return "<1%" if p < 0.005 else f"{int(round(p * 100))}%"


def hand_sentence(facts):
    """Plain facts about the user's hand, for a position with nothing measured to quote.

    'You have 13 HCP and a balanced hand (4♠ 3♥ 3♦ 3♣).'
    """
    shape = " ".join(f"{n}{'♠♥♦♣'[i]}" for i, n in enumerate(facts["shape"]))
    if facts["balanced"]:
        return f"You have {facts['hcp']} HCP and a balanced hand ({shape})."
    longest = facts["longest"]
    held = facts["shape"][SUITS.index(longest)]
    return (f"You have {facts['hcp']} HCP and an unbalanced hand ({shape}); your longest "
            f"suit is {cards_words(held, longest)}.")


def net_sentence(facts, net):
    """The net's own choice on these exact cards, said to a beginner.

    'You have 13 HCP and 5 hearts: with exactly these cards the net bids 1♥ 78% of
    the time.' The hand clause names only what bears on the call - the suit it
    names, or the shape for notrump - and the percentage is the net's policy.
    """
    if not net or not net["calls"]:
        return None
    top = net["calls"][0]
    name = top["call"]
    hand = f"You have {facts['hcp']} HCP"
    if name.endswith("NT"):
        hand += " and a balanced hand" if facts["balanced"] else " and an unbalanced hand"
    elif name not in ("Pass", "X", "XX"):
        hand += f" and {cards_words(facts['shape'][SUITS.index(name[-1])], name[-1])}"
    if top["p"] is None:
        return f"{hand}: the net's best-scoring call here is {call_glyph(name)}."
    verb = {"Pass": "passes", "X": "doubles", "XX": "redoubles"}.get(
        name, f"bids {call_glyph(name)}")
    out = f"{hand}: with exactly these cards the net {verb} {percent_words(top['p'])} of the time."
    runner = net["calls"][1] if len(net["calls"]) > 1 else None
    if runner and runner["p"] is not None and runner["p"] >= RUNNER_UP_AT:
        out += f" Its next choice is {call_glyph(runner['call'])} ({percent_words(runner['p'])})."
    return out


def corpus_rows(entry, legal, facts):
    """One row per call self-play made at a position, most frequent first."""
    rows = []
    for token, stats in sorted(entry["calls"].items(), key=lambda kv: -kv[1]["share"]):
        action = token_to_call(token)
        name = call_name(action)
        rows.append({"call": name, "action": action, "token": token,
                     "stats": stats, "legal": bool(legal[action]),
                     "fit": fits_hand(stats, facts),
                     "says": would_show_sentence(name, stats),
                     "yours": fit_sentence(stats, facts)})
    return rows


NEAREST_KEEP = 4         # a front-trimmed stand-in keeps at least one whole round


def nearby_auctions(calls, me, dealer=DEALER):
    """Stand-ins for an auction self-play never reached, most faithful first.

    Each is (calls, what changed). Every one keeps the user on turn next and
    partner two calls back, so only the calls' history changes, never who is who:

    - the opening passes left out: the same auction opened in an earlier seat;
    - the opponents' bids and doubles replaced by passes, most recent first, then
      all of them: the same conversation between the user and partner;
    - the earliest calls dropped, keeping at least a whole round.
    """
    words = lambda cs: " ".join(call_glyph(call_name(c)) for c in cs)
    lead = next((i for i, c in enumerate(calls) if c != PASS), len(calls))
    for k in range(1, lead + 1):
        yield calls[k:], "the same calls with the opening passes left out"
    theirs = [i for i, c in enumerate(calls) if c != PASS and (dealer + i) % 2 != me % 2]
    for i in reversed(theirs):
        yield (calls[:i] + [PASS] + calls[i + 1:],
               f"the same, but with the opponents' {words([calls[i]])} replaced by a pass")
    if len(theirs) > 1:
        yield ([PASS if i in theirs else c for i, c in enumerate(calls)],
               f"the same, but with all the opponents' calls ({words(calls[i] for i in theirs)}) "
               f"replaced by passes")
    for k in range(1, len(calls) - NEAREST_KEEP + 1):
        first = "the first call" if k == 1 else f"the first {k} calls"
        yield (calls[k:],
               f"the last {len(calls) - k} calls only, ignoring {first} ({words(calls[:k])})")


def nearest_position(game, legal, facts):
    """The closest auction self-play did reach, for a position it never did.

    The corpus is keyed by the whole auction from the dealer, stops eight calls
    deep, and leaves out any position fewer than 200 hands reached. The first
    stand-in from `nearby_auctions` that the corpus holds is used, and the note
    says exactly what was changed to get there. Only calls legal here are kept.
    """
    table = corpus_table(game)
    if table is None:
        return None
    for calls, change in nearby_auctions(game["calls"], game["user_seat"], game["dealer"]):
        if not any(c != PASS for c in calls):
            continue                          # all passes: nothing left that says anything
        key = "-".join(call_token(c) for c in calls)
        entry = table.get(key)
        if entry is None:
            continue
        shown = " ".join(call_glyph(call_name(c)) for c in calls)
        return {"position": key, "n": entry["n"], "change": change,
                "note": f"Self-play never reached this exact auction. The nearest it did "
                        f"reach is {shown} — {change}.",
                "rows": [r for r in corpus_rows(entry, legal, facts) if r["legal"]]}
    return None


def auction_hint(game):
    """Everything the teaching panel shows before the user calls."""
    view, legal = bid_view(game)
    facts = hand_facts(game["bitmaps"][game["user_seat"]])
    here = corpus_at(game, game["calls"])
    net = None
    if view is not None and view.get("policy"):
        ranked = sorted(((p, c) for c, p in enumerate(view["policy"]) if legal[c]),
                        reverse=True)
        q = view.get("q") or []
        net = {
            "calls": [{"call": call_name(c), "action": c, "p": round(float(p), 4),
                       "q": (round(float(q[c]) * 100, 0) if q else None)}
                      for p, c in ranked[:3]],
            # Every legal call's share, so hovering any call in the box can quote it.
            "p_by_call": {str(c): round(float(p), 4) for p, c in ranked},
            "note": "Its chance for each call with your exact cards, and how many points "
                    "below the best possible contract it expects to finish after each "
                    "(0 is perfect, less is worse).",
        }
    elif view is not None:
        best = max((view["q"][c], c) for c in range(N_CALLS) if legal[c])
        net = {"calls": [{"call": call_name(best[1]), "action": best[1], "p": None,
                          "q": round(float(best[0]) * 100, 0)}],
               "p_by_call": {},
               "note": "This model scores calls instead of ranking them; here is its best."}

    rows = corpus_rows(here, legal, facts) if here else []
    candidates = [r["action"] for r in (net["calls"] if net else [])][:2]
    replies = []
    for c in candidates:
        r = partner_reply(game, c)
        if r is not None:
            replies.append({"call": call_name(c), **r})
    # One line of advice: what the suggested call would say, when the corpus knows it.
    # Where self-play never reached this auction, the nearest one it did reach
    # stands in, labelled as the approximation it is.
    pick = net["calls"][0]["action"] if net else None
    nearest = None if here else nearest_position(game, legal, facts)
    picked = next((r for r in rows if r["action"] == pick), None)
    near = next((r for r in (nearest or {}).get("rows", []) if r["action"] == pick), None)
    return {
        "kind": "auction",
        "seat": game["user_seat"],
        "net": net,
        "net_says": net_sentence(facts, net),
        "hand_says": hand_sentence(facts),
        "advice": (picked or {}).get("says"),
        "advice_fit": (picked or {}).get("yours"),
        "nearest": nearest,
        "nearest_advice": (near or {}).get("says"),
        "nearest_fit": (near or {}).get("yours"),
        "told": recent_calls(game),
        "position": "-".join(call_token(c) for c in game["calls"]) or "(opening)",
        "corpus_n": here["n"] if here else None,
        "corpus": rows,
        "partner_last": partner_last(game),
        "partner_next": replies,
        "your_hand": facts,
        "has_corpus": corpus_table(game) is not None,
        "rule": rule_hint(game, net),
        "conditions_note": conditions_note(game),
    }


def conditions_note(game):
    """The self-play table and the rules were measured with nobody vulnerable. Dealer is
    no matter: both are keyed by the calls from the dealer and seats relative to it, as
    the nets see them. Vulnerability they never saw, so say so when it is not 'none'."""
    if game["vul"] == (False, False):
        return None
    return (f"This board is {VUL_NAMES[game['vul']]}. The self-play meanings and the rule of "
            f"thumb were measured with nobody vulnerable; the net's own numbers use the real "
            f"vulnerability.")


# ------------------------------------------------- rule of thumb (experimental)
# D's 15 teaching rules, a frozen copy of bridge_new/experiments/teaching_D
# (emergent/teaching/: situations.py + rules.json, numpy only). A separate
# source from `net` / `corpus`: a short rule distilled from D's self-play
# (dealer North, nobody vulnerable -- the table's own setting), with how often D
# itself makes the call the rule names, measured on held-out self-play.

TEACH_RULES = teaching.load_rules()
TEACH_MODEL = "D_cw_s75k"           # the net the rules were distilled from
TEACH_ORDER = ["R1", "R2", "O1", "O2", "O3", "O4", "D1", "D3", "D4",
               "C1", "P1", "P2", "P3", "G1", "G2"]          # README's teaching order
ACT_WORDS = {"pass": "pass", "double": "double", "redouble": "redouble",
             "support": "raise", "rebid": "rebid own suit", "new_suit": "new suit",
             "nt": "notrump", "their_suit": "their suit"}
KIND_WORDS = {"min": "", "jump": " (jump)", "game": " (game)", "slam": " (slam)"}


def action_words(label):
    """'support_min' -> 'raise', 'new_suit_jump' -> 'new suit (jump)'."""
    for t, w in ACT_WORDS.items():
        if label == t:
            return w
        if label.startswith(t + "_"):
            return w + KIND_WORDS.get(label[len(t) + 1:], " " + label[len(t) + 1:])
    return label.replace("_", " ")


def rule_hint(game, net):
    """The rule of thumb that owns this decision, or a note that none does.

    Never touches the other hint fields; `net_call` is copied from `net` only so
    the page can say, neutrally, when the rule and the net disagree.
    """
    seat = game["user_seat"]
    cards = [c for c in range(52) if int(game["owners"][c]) == seat]
    net_top = net["calls"][0] if net and net.get("calls") else None
    base = {"model": game["model"], "rules_model": TEACH_MODEL,
            "model_note": (None if game["model"] == TEACH_MODEL else
                           f"These rules describe D ({TEACH_MODEL}); the bidding model at "
                           f"this table is {game['model']}."),
            "net_call": call_glyph(net_top["call"]) if net_top else None,
            "net_call_name": net_top["call"] if net_top else None,
            "net_action": net_top["action"] if net_top else None}
    try:
        out = teaching.advise(list(game["calls"]), game["dealer"], seat, cards, TEACH_RULES)
    except Exception as e:                       # experimental: never break the hint
        return {**base, "covered": False, "note": f"The rule of thumb failed here ({e})."}
    if out is None:
        return {**base, "covered": False,
                "note": "The rules of thumb start after the opening bid; opening bids are "
                        "not covered."}
    r, b = out["rule"], out["branch"]
    bs, rs = b.get("stats") or {}, r.get("stats") or {}
    f = out["facts"]
    call = int(out["call"])
    lines = [x for x in r["branches"] if (x.get("stats") or {}).get("share")]
    return {
        **base,
        "covered": True,
        "id": r["id"],
        "number": TEACH_ORDER.index(r["id"]) + 1 if r["id"] in TEACH_ORDER else None,
        "title": r["title"],
        "when": r["when"],
        "think": r["think"],
        "line": b["say"],
        "line_no": (lines.index(b) + 1) if b in lines else None,
        "suggested": call_glyph(call_name(call)),
        "suggested_name": call_name(call),
        "suggested_action": call,
        "same_as_net": net_top is not None and net_top["action"] == call,
        # measured on held-out self-play of D: share of all post-opening decisions,
        # and how often D made exactly the call this line / this rule names
        "agree_line": bs.get("agree_call"),
        "line_share": bs.get("share"),
        "d_does": [{"what": action_words(k), "share": v} for k, v in (bs.get("d_does") or {}).items()][:3],
        "agree_rule": rs.get("agree_call"),
        "agree_rule_d_sure": rs.get("agree_when_d_sure"),
        "notes": r.get("notes") or None,
        "sayc_rule": r.get("sayc"),
        "sayc_line": b.get("sayc"),
        "facts": {"hcp": f["hcp"], "partner_mid": f["partner_mid"] if f["partner_bids"] else None,
                  "total": f["total"] if f["partner_bids"] else None,
                  "support": f["support"] if f["partner_suit"] else None,
                  "fit_cards": f["fit_cards"] if f["partner_suit"] else None},
    }


# -------------------------------------------------------------- play teaching

def seen_cards(game, seen):
    """Every card the user can account for: their own hands, and everything played."""
    known = np.zeros(52, dtype=bool)
    for c in range(52):
        if int(game["owners"][c]) in seen:
            known[c] = True
    for c in game["pg"]["played"]:
        known[c] = True
    return known


def position_facts(game, batch, turn, seen):
    """Plain statements about the position, each read straight off the cards.

    Every line here is arithmetic on what the user can already see. Nothing
    guesses at a hidden hand and nothing quotes the solver.
    """
    pg = game["pg"]
    trump = pg["trump"]
    declarer, dummy = declarer_dummy(game)
    facts = []
    add = facts.append

    trick = [int(c) for c in batch.trick_cards()[0].tolist()]
    pos = len(trick)
    my_cards = [c for c in range(52) if int(game["owners"][c]) == turn and batch.unplayed[0, c]]

    if pos == 0:
        if batch.trick_no == 0:
            label = f"{pg['level']}{'♠♥♦♣'[trump] if trump < 4 else 'NT'}"
            add(f"This is the opening lead against {label}"
                f"{['', ' doubled', ' redoubled'][pg['doubled']]} by "
                f"{SEAT_NAMES[declarer]}. Dummy is still face down — you lead blind.")
        else:
            add(f"You won the last trick, so you lead to trick {batch.trick_no + 1}.")
    else:
        led = trick[0] // 13
        best_i = trick_best(trick, trump)
        leader = int(batch.leader[0])
        winner_seat = (leader + best_i) % 4
        ordinal = ["first", "second", "third", "last"][pos]
        add(f"{SEAT_NAMES[leader]} led {card_text(trick[0])}. You play {ordinal} to this trick"
            + (", so you already know what it takes to win it." if pos == 3 else "."))
        partner = "dummy, your partner's hand" if winner_seat == dummy else "your partner"
        rel = partner if winner_seat % 2 == turn % 2 else "an opponent"
        add(f"{card_text(trick[best_i])} from {SEAT_NAMES[winner_seat]} is winning so far — "
            f"that is {rel}.")
        mine_in_suit = [c for c in my_cards if c // 13 == led]
        if mine_in_suit:
            add(f"You hold {len(mine_in_suit)} {SUIT_WORDS[SUITS[led]]}"
                f"{'' if len(mine_in_suit) == 1 else 's'}, so you must follow suit.")
        else:
            trumps = [c for c in my_cards if c // 13 == trump] if trump < 4 else []
            if trumps:
                add(f"You have no {SUIT_WORDS[SUITS[led]]}s left, so you may ruff with one of "
                    f"your {len(trumps)} trump{'' if len(trumps) == 1 else 's'} or throw "
                    f"something away.")
            else:
                add(f"You have no {SUIT_WORDS[SUITS[led]]}s left and nothing to ruff with, "
                    f"so this card is a discard.")
        legal_now = [c for c in my_cards if batch.legal()[0, c]]
        winners = [c for c in legal_now if beats(c, trick[best_i], trump)]
        left = 3 - pos
        if pos == 3:
            add(f"{len(winners)} of your cards take{'s' if len(winners) == 1 else ''} this trick."
                if winners else "Nothing you hold beats it; this trick is gone.")
        elif winners:
            add(f"{len(winners)} of your cards would be on top for now, but "
                f"{left} player{'' if left == 1 else 's'} still "
                f"{'plays' if left == 1 else 'play'} after you.")

    if trump < 4:
        known = seen_cards(game, seen)
        out = sum(1 for c in range(trump * 13, trump * 13 + 13) if not known[c])
        mine = sum(1 for c in my_cards if c // 13 == trump)
        if turn == declarer and dummy in seen:
            ours = sum(1 for c in range(52) if int(game["owners"][c]) in (declarer, dummy)
                       and batch.unplayed[0, c] and c // 13 == trump)
            add(f"Trumps: you and dummy still hold {ours}; {out} you cannot see are still out.")
        else:
            add(f"Trumps: you hold {mine}; {out} that you cannot see are still out.")
        if out:
            led_trump = 0
            for t in range(batch.trick_no):
                first = int(batch.history[0, t * 4])
                lead_seat = int(batch.trick_winner[0, t - 1]) if t else (declarer + 1) % 4
                if first // 13 == trump and lead_seat % 2 == declarer % 2:
                    led_trump += 1
            if led_trump >= 2:
                add(f"The declaring side has led trumps on {led_trump} of the "
                    f"{batch.trick_no} tricks so far and {out} are still out — that is "
                    f"drawing trumps.")
    return facts


def finesse_note(game, batch, turn, seen):
    """A finesse, only when the cards on the table prove one is available.

    Declarer sees both of the declaring hands, so this is the one place the shape
    of the play is fully checkable: leading a low card from one hand towards an
    honour in the other, with a higher card still out, wins exactly when the
    player sitting in between holds that higher card. Stated as a fact about the
    cards, never as the net's reason.
    """
    declarer, dummy = declarer_dummy(game)
    if turn != declarer or game["user_seat"] != declarer:
        return None                                   # only declarer sees both hands
    if len(batch.trick_cards()[0].tolist()):
        return None                                   # only when leading to a trick
    partner_hand = dummy
    if partner_hand not in seen:
        return None                                   # dummy is not down yet
    victim = (turn + 1) % 4                           # plays between the two hands
    known = seen_cards(game, seen)
    mine = [c for c in range(52) if int(game["owners"][c]) == turn and batch.unplayed[0, c]]
    theirs = [c for c in range(52) if int(game["owners"][c]) == partner_hand
              and batch.unplayed[0, c]]
    for suit in range(4):
        here = sorted(c for c in mine if c // 13 == suit)
        over = sorted(c for c in theirs if c // 13 == suit)
        if not here or not over:
            continue
        low = here[-1]
        honour = missing = None
        for cand in over:
            if cand % 13 > 3:                          # only A K Q J are worth finessing for
                continue
            gap = [c for c in range(suit * 13, cand) if not known[c]]
            if gap and low % 13 > cand % 13:           # and we must be leading the lower card
                honour, missing = cand, gap
                break
        if honour is None:
            continue
        return {
            "suit": SUITS[suit],
            "lead": low, "lead_name": card_text(low),
            "honour": honour, "honour_name": card_text(honour),
            "missing": [card_text(c) for c in missing[:2]],
            "victim": SEAT_NAMES[victim],
            "text": f"Leading {card_text(low)} towards {card_text(honour)} in the other hand "
                    f"is a finesse: it wins if {SEAT_NAMES[victim]} holds "
                    f"{' or '.join(card_text(c) for c in missing[:2])}, because "
                    f"{SEAT_NAMES[victim]} has to play before it.",
        }
    return None


def book_advice(game, batch, turn):
    """Ordinary club advice for the opening lead. Ours, not the net's, and labelled so.

    Only for the lead: it is the one moment where a single sentence is genuinely
    standard. After that the position decides, and a generic rule would be worse
    than nothing.
    """
    pg = game["pg"]
    if batch.trick_no or len(batch.trick_cards()[0].tolist()) or turn == pg["declarer"]:
        return None
    if pg["trump"] == 4:
        return ("Against notrump the usual opening lead is the fourth highest card of your "
                "longest, strongest suit — you are trying to set up small cards before "
                "declarer sets up theirs.")
    return ("Against a suit contract the usual opening leads are the top of touching honours, "
            "or a singleton if you hope to ruff. Leading away from an ace is the classic "
            "way to lose one.")


def card_notes(game, batch, turn, seen, card):
    """Facts about one card. Properties of the card, not motives for playing it."""
    pg = game["pg"]
    trump = pg["trump"]
    notes = []
    suit = card // 13
    my_cards = sorted(c for c in range(52) if int(game["owners"][c]) == turn
                      and batch.unplayed[0, c])
    in_suit = [c for c in my_cards if c // 13 == suit]
    trick = [int(c) for c in batch.trick_cards()[0].tolist()]
    known = seen_cards(game, seen)

    if len(in_suit) > 1:
        if card == in_suit[-1]:
            notes.append(f"your lowest {SUIT_WORDS[SUITS[suit]]}")
        elif card == in_suit[0]:
            notes.append(f"your highest {SUIT_WORDS[SUITS[suit]]}")
    # Touching honours, top first. Card index rises as rank falls, so card + 1 is
    # the next card down.
    if card % 13 <= 4 and (card - 1) not in in_suit and (card + 1) in in_suit:
        seq = [card]
        while seq[-1] + 1 in in_suit:
            seq.append(seq[-1] + 1)
        notes.append("the top of your touching "
                     + "".join(rank_text(c) for c in seq))
    higher_out = [c for c in range(suit * 13, card) if not known[c]]
    if not higher_out and card % 13 <= 6:
        if trump < 4 and suit != trump:
            notes.append(f"the best {SUIT_WORDS[SUITS[suit]]} left — only a trump beats it now")
        else:
            notes.append(f"a sure winner: every higher {SUIT_WORDS[SUITS[suit]]} has been "
                         f"played or sits in a hand you can see")
    if trick:
        led = trick[0] // 13
        best_i = trick_best(trick, trump)
        if suit != led and trump < 4 and suit == trump:
            notes.append("a ruff")
        elif suit != led:
            notes.append("a discard")
        if beats(card, trick[best_i], trump):
            notes.append("the card that takes the trick" if len(trick) == 3
                         else "winning the trick as it stands")
    return notes


def belief_highlights(view, top=6):
    """The net's belief head, cut down to the cards a club player cares about.

    `playdesk.belief_view` already renormalises over the hands that are actually
    possible; this keeps the honours and drops the true holder, which the user is
    not allowed to see yet.
    """
    bel = view.get("belief") or {}
    rows = []
    for c in bel.get("cards", []):
        rank = c["card"] % 13
        if rank > 2:                                    # aces, kings and queens only
            continue
        best = max(c["p"].items(), key=lambda kv: kv[1])
        rows.append({"card": c["card"], "name": card_text(c["card"]),
                     "seat": int(best[0]), "seat_name": SEAT_NAMES[int(best[0])],
                     "p": round(float(best[1]), 3),
                     "spread": {k: round(float(v), 3) for k, v in c["p"].items()}})
    rows.sort(key=lambda r: (r["card"] % 13, -r["p"]))
    return {"cards": rows[:top], "seats": bel.get("seats", []),
            "hidden": bel.get("hidden", 0)}


def solver_block(game, batch):
    """Double dummy at this exact position. Off unless the user asks; always labelled."""
    if playdesk.SOLVER is not None:
        return {"available": False, "reason": playdesk.SOLVER}
    try:
        per_card = playdesk.solve_here(playdesk.solver_deal(game["pg"]))
    except Exception as exc:                            # keep the desk alive
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if not per_card:
        return {"available": False, "reason": "nothing to solve here"}
    best = max(per_card.values())
    return {"available": True, "cards": {str(k): v for k, v in per_card.items()},
            "best": sorted(k for k, v in per_card.items() if v == best),
            "best_tricks": best,
            "side": "declaring" if int(batch.to_play()[0]) % 2 == game["pg"]["declarer"] % 2
                    else "defending"}


def play_advice(net):
    """One line of advice: the net's card, how sure it is, and one fact about the card.

    The notes are facts about the card, so the sentence never says *why* it chose
    it, only what the card is. The percentages are its policy on the user's own
    information, and a close second choice is named, so a coin flip never reads
    like a certainty.
    """
    card = card_text(net["pick"])
    head = f"The net plays {card} here {percent_words(net['confidence'])} of the time"
    out = f"{head} — it is {net['notes'][0]}." if net["notes"] else f"{head}."
    runner = next((t for t in net["top"] if t["card"] != net["pick"]), None)
    if runner and runner["p"] >= RUNNER_UP_AT:
        out += f" Its next choice is {runner['name']} ({percent_words(runner['p'])})."
    return out


def play_hint(game):
    pg = game["pg"]
    contracts = playdesk.contracts_of(pg)
    batch = batch_for(game)
    turn = int(batch.to_play()[0])
    seen = known_seats(game)
    bot = play_models().get(game["play_model"])
    view = None if bot is None else bot.view(contracts, batch, len(pg["played"]))
    out = {
        "kind": "play",
        "seat": turn, "seat_name": SEAT_NAMES[turn],
        "facts": position_facts(game, batch, turn, seen),
        "finesse": finesse_note(game, batch, turn, seen),
        "book": book_advice(game, batch, turn),
        "net": None, "belief": None, "solver": None,
    }
    if view is not None:
        top = [t for t in view["top"]][:3]
        out["net"] = {
            "pick": view["pick"], "pick_name": card_text(view["pick"]),
            "confidence": round(float(view["confidence"]), 3),
            "top": [{"card": t["card"], "name": card_text(t["card"]),
                     "p": round(float(t["p"]), 4)} for t in top],
            "notes": card_notes(game, batch, turn, seen, view["pick"]),
            "probs": view["probs"],
        }
        out["belief"] = belief_highlights(view)
        out["advice"] = play_advice(out["net"])
    out["context"] = out["facts"][0] if out["facts"] else None
    if game["peek"]:
        out["solver"] = solver_block(game, batch)
    return out


def suggest_from_hint(hint):
    """The move the Next button commits, taken off the hint's own forward pass.

    Only ever built with hints on: with them off the user picks every call and
    card unaided, and the payload says nothing about what the nets would do.
    """
    if not hint or not hint.get("net"):
        return None
    if hint["kind"] == "auction":
        calls = hint["net"].get("calls") or []
        if not calls:
            return None
        return {"kind": "call", "action": calls[0]["action"],
                "label": call_glyph(calls[0]["call"]), "name": calls[0]["call"]}
    pick = hint["net"]["pick"]
    return {"kind": "card", "action": int(pick), "label": card_text(pick),
            "name": card_name(pick)}


def user_review(game):
    """After the thirteenth trick: the user's own cards that gave a trick away.

    `playdesk.review` re-solves the board card by card, so every line here is the
    solver's, not ours. Only the cards the user actually chose are kept.
    """
    rows = playdesk.review(game["pg"])
    if rows is None:
        return None
    # Cards after a claim were the solver's, not the user's.
    upto = game["claim"]["at"] if game.get("claim") else 52
    yours = [r for r in rows["cards"] if user_plays(game, r["seat"]) and r["step"] < upto]
    return {
        "available": True,
        "mine": [{**r, "name": card_text(r["card"]),
                  "best_cards": [card_text(c) for c in r["best_cards"]]}
                 for r in yours if r["lost"] > 0],
        "played": len(yours),
        "lost": sum(r["lost"] for r in yours),
        "declaring_lost": rows["declaring_lost"], "defending_lost": rows["defending_lost"],
        "note": "Each card was re-solved with all four hands face up, against a defence "
                "that never errs. Real opponents do err, so a card marked here is not "
                "always a mistake at the table.",
    }


# --------------------------------------------------------- the "where it goes" job

def _explain_worker():
    while True:
        with EXPLAIN["lock"]:
            job = EXPLAIN["next"]
            EXPLAIN["next"] = None
            if job is None:
                EXPLAIN["worker"] = None
                return
            key, snapshot = job
            EXPLAIN["jobs"][key] = {"status": "running"}
        try:
            from emergent.explain import explain as run_explain
            bot = bid_models()[snapshot["model"]]
            result = run_explain(bot, snapshot, snapshot["candidates"], snapshot["samples"])
            d = snapshot["dealer"]
            for p in result["picture"]:
                p["seat"] = (p["seat"] + d) % 4
            for c in result["candidates"]:
                for line in c["lines"]:
                    if line["declarer"] is not None:
                        line["declarer"] = (line["declarer"] + d) % 4
            entry = {"status": "done", "result": result}
        except Exception as exc:                        # keep the desk alive on any failure
            entry = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        with EXPLAIN["lock"]:
            EXPLAIN["jobs"][key] = entry
            while len(EXPLAIN["jobs"]) > MAX_EXPLAIN_JOBS:
                EXPLAIN["jobs"].popitem(last=False)


# ----------------------------------------------------------------- shared links
#
# A link carries the whole position, in the bid desk's and play desk's own codecs:
#
#     #d=<deal>&a=<calls>&dr=0&v=none&p=<cards>&s=<N|E|S|W>&m=<bid model>&pm=<play model>&h=<0|1>
#
# `d` is `bidserver.encode_deal`, `a` one `bidserver.CALL_CHARS` character per call,
# `p` one `playdesk.CARD_CHARS` character per card in play order, `m` the bid desk's
# model key. So the same link opens on `/` (the auction) and `/play` (the cards).
# The deal is all four hands: it has to be, to rebuild the board. The page still
# only draws what the chair in `s` may see.

SEAT_LETTERS = "NESW"


def link_code(game):
    """The pieces of the shared link for the position the server holds.

    The page cuts `a` and `p` back to the frame it is drawing (one character per
    call and per card), so a link copied mid-animation is the position on screen.
    """
    pg = game["pg"]
    return {
        "d": DESK.encode_deal(game["owners"]) if DESK is not None else None,
        "a": "".join(DESK.CALL_CHARS[c] for c in game["calls"]) if DESK is not None else "",
        "p": playdesk.encode_cards(pg["played"]) if pg is not None else "",
        "dr": game["dealer"], "v": playdesk.vul_code(*game["vul"]),
        "s": SEAT_LETTERS[game["user_seat"]],
        "m": game["model"], "pm": game["play_model"],
        "h": int(bool(game["hints"])),
    }


class LinkError(ValueError):
    pass


LINK_VULS = {"none": (False, False), "ns": (True, False), "ew": (False, True),
             "all": (True, True), "both": (True, True)}


def link_text(body, key, limit):
    """One field of a link body as a string ("" when missing), or `LinkError`."""
    v = body.get(key, "")
    if v is None:
        v = ""
    if not isinstance(v, (str, int)) or isinstance(v, bool):
        raise LinkError(f"bad {key} in link")
    v = str(v)
    if len(v) > limit:
        raise LinkError(f"{key} in link is too long")
    return v


def board_from_link(body, prev):
    """A fresh game dict at exactly the position in `body`, or `LinkError`.

    Everything is checked: the deal decodes to 13 cards a hand, the chair is one of
    N/E/S/W, the models exist, every call is legal in turn (by the rules, not by
    what the current bidding net may say), and every card is legal in turn once
    the auction has produced a contract. Nothing is acted on past the last call or
    card in the link, even when a net is on turn there: the caller does not
    `advance`, and the page waits for the visitor before letting the nets go on.
    """
    def text(key, limit):
        return link_text(body, key, limit)

    code = text("deal", 64)
    owners = DESK.decode_deal(code) if (code and DESK is not None) else None
    if owners is None:
        raise LinkError("bad deal in link: it must be the 18-character code of a full deal")
    seat = text("seat", 1).upper()
    if seat == "" or seat not in SEAT_LETTERS:
        raise LinkError("bad seat in link: it must be N, E, S or W")
    dealer = text("dealer", 2) or "0"
    if dealer not in ("0", "1", "2", "3"):
        raise LinkError("bad dealer in link: dr must be 0, 1, 2 or 3 (North, East, South, West)")
    vul = text("vul", 8) or "none"
    if vul not in LINK_VULS:
        raise LinkError("bad vulnerability in link: v must be none, ns, ew or all")
    model, play_model = text("model", 64), text("play_model", 64)
    if model and model not in bid_models():
        raise LinkError(f"bidding model {model!r} is not on this server")
    if play_model and play_model not in play_models():
        raise LinkError(f"card-play model {play_model!r} is not on this server")
    hints = body.get("hints")
    hints = prev["hints"] if hints is None or hints == "" else str(hints) in ("1", "true", "True")

    game = new_board(user_seat=SEAT_LETTERS.index(seat),
                     model=model or prev["model"], play_model=play_model or prev["play_model"],
                     hints=hints, peek=prev["peek"], owners=owners,
                     board_no=prev["board_no"] + 1, dealer=int(dealer), vul=LINK_VULS[vul])
    game["search"] = prev.get("search", engine.CONFIG.search)

    replay_moves(game, text("auction", 400), text("played", 64))
    return game


def replay_moves(game, auction, cards):
    """Play `auction` (one CALL_CHARS character a call) and `cards` (one CARD_CHARS
    character a card) onto a fresh game, checking each is legal in turn."""
    for i, ch in enumerate(auction):
        call = DESK.CALL_CHARS.find(ch)
        st = auction_of(game)
        if st.ended:
            raise LinkError(f"call {i + 1} in link comes after the auction is over")
        if not 0 <= call < N_CALLS:
            raise LinkError(f"call {i + 1} in link ({ch!r}) is not a call")
        if not bool(st.legal_mask()[call]):
            raise LinkError(f"call {i + 1} in link, {call_name(call)} by "
                            f"{SEAT_NAMES[st.turn]}, is not legal there")
        game["calls"].append(call)
        if auction_of(game).ended:
            begin_play(game)

    if cards:
        st = auction_of(game)
        if not st.ended:
            raise LinkError("the link has cards played before the auction is over")
        if game["pg"] is None:
            raise LinkError("the link has cards played on a passed-out board")
    for i, ch in enumerate(cards):
        card = playdesk.CARD_CHARS.find(ch)
        if card < 0:
            raise LinkError(f"card {i + 1} in link ({ch!r}) is not a card")
        batch = batch_for(game)
        if batch.done:
            raise LinkError(f"card {i + 1} in link comes after the thirteenth trick")
        if not playdesk.play_card(game["pg"], card, batch):
            raise LinkError(f"card {i + 1} in link, {card_name(card)} by "
                            f"{SEAT_NAMES[int(batch.to_play()[0])]}, is not legal there")
        game.pop("_batch", None)


# -------------------------------------------------------------------- /debug
#
# The table with all four hands face up and the position in the address bar in
# words a person can read and type, so a board can be set up by hand and copied
# at any point:
#
#   /debug#n=AQJ32.Q8.T32.K83&e=T54.KT63.875.AJ9&s=K987.AJ.AKQJ.QT4&w=6.97542.964.7652
#          &dealer=E&vul=ns&auction=1S-P-2C-P&play=HK-HA-H2-H3&seat=S&bm=<bid model>&pm=<play model>
#
# - `n` `e` `s` `w`  hands as spades.hearts.diamonds.clubs, ranks AKQJT98765432
#                    (10 reads as T), `-` or nothing for a void. A hand left out is
#                    dealt at random from the cards nobody holds; all four out is a
#                    random deal.
# - `dealer`         N, E, S or W (default N).   `vul`  none, ns, ew or both (default none).
# - `auction`        calls from the dealer joined by `-`: P, X, XX, 1C ... 7NT (1N works too).
# - `play`           cards in play order joined by `-`, suit then rank: HK-HA-H2-H3.
# - `seat`, `bots`   who plays what. Neither: you call and play for all four chairs and
#                    the nets only move when asked (a step, or play to the end); South
#                    sits at the bottom. `seat=S`: you hold South and the nets the other
#                    three, as on /. `bots=EW` (chairs the nets play; `-` for none) with
#                    `seat` as the bottom chair: any mix.
# - `bm`, `pm`       the bidding and card-play model ids (models/models.json,
#                    models/play_models.json). Left out: the server's defaults, so a
#                    link means the same models in any tab; the page leaves them out
#                    when they are the defaults. An old link's `m` reads as `bm`.
# - `hints`          1 or 0, the hints toggle (left out: as the tab had it).
#
# Everything is checked like a link on /, and the board stops at the position the
# same way. The hints read only what the user's chair may see, as on /. A debug game
# never starts a challenge, so it cannot reach the rating or the leaderboard.

DEBUG_VULS = {"none": (False, False), "-": (False, False), "ns": (True, False),
              "ew": (False, True), "both": (True, True), "all": (True, True)}
DEBUG_VUL_CODES = {v: k for k, v in DEBUG_VULS.items() if k in ("none", "ns", "ew", "both")}
CALL_WORDS = {**{n: i for i, n in enumerate(NAMES)},
              **{n[:-1]: i for i, n in enumerate(NAMES) if n.endswith("NT")},
              "P": PASS, "PASS": PASS, "X": DOUBLE, "D": DOUBLE, "DBL": DOUBLE,
              "XX": REDOUBLE, "R": REDOUBLE, "RDBL": REDOUBLE}


def parse_hand(text, seat):
    """'AQJ32.Q8.T32.K83' -> 13 card indices, or `LinkError` naming what is wrong."""
    name = SEAT_NAMES[seat]
    parts = text.strip().upper().replace("10", "T").split(".")
    if len(parts) != 4:
        raise LinkError(f"{name}'s hand needs four suits, spades.hearts.diamonds.clubs "
                        f"(like AQJ32.Q8.T32.K83), not {text!r}")
    cards = []
    for s, part in enumerate(parts):
        for ch in ("" if part == "-" else part):
            r = RANKS.find(ch)
            if r < 0:
                raise LinkError(f"{name}'s {SUIT_WORDS[SUITS[s]]}s: {ch!r} is not a rank "
                                f"(AKQJT98765432, - for a void)")
            if s * 13 + r in cards:
                raise LinkError(f"{name} holds the {card_text(s * 13 + r)} twice")
            cards.append(s * 13 + r)
    if len(cards) != 13:
        raise LinkError(f"{name} has {len(cards)} cards, not 13")
    return cards


def parse_tokens(text, what, lookup):
    """'1S-P-2C' -> indices through `lookup(token)`, or `LinkError` naming the bad one."""
    out = []
    for i, tok in enumerate(t for t in text.replace(",", "-").replace(" ", "-").split("-") if t):
        v = lookup(tok.upper())
        if v is None:
            raise LinkError(f"{what} {i + 1} ({tok!r}) is not a "
                            + ("call: use P, X, XX or 1C ... 7NT" if what == "call"
                               else "card: use suit then rank, like HK or D10"))
        out.append(v)
    return out


def card_word(tok):
    tok = tok.replace("10", "T")
    if len(tok) != 2 or tok[0] not in SUITS or tok[1] not in RANKS:
        return None
    return SUITS.index(tok[0]) * 13 + RANKS.index(tok[1])


def debug_board(body, prev):
    """A debug game dict at exactly the position in `body` (the readable link), or `LinkError`."""
    hands, held = {}, {}
    for seat, key in enumerate("nesw"):
        text = link_text(body, key, 40)
        if not text:
            continue
        hands[seat] = parse_hand(text, seat)
        for c in hands[seat]:
            if c in held:
                raise LinkError(f"the {card_text(c)} is in both {SEAT_NAMES[held[c]]}'s "
                                f"and {SEAT_NAMES[seat]}'s hands")
            held[c] = seat
    owners = np.full(52, -1, dtype=np.int64)
    for seat, cards in hands.items():
        owners[cards] = seat
    rest = [s for s in range(4) if s not in hands]          # deal what nobody holds
    free = np.flatnonzero(owners < 0)
    owners[np.random.default_rng().permutation(free)] = np.repeat(rest, 13)

    dealer = link_text(body, "dealer", 5).upper() or "N"
    if dealer not in SEAT_LETTERS or len(dealer) != 1:
        raise LinkError("bad dealer: it must be N, E, S or W")
    vul = link_text(body, "vul", 5).lower() or "none"
    if vul not in DEBUG_VULS:
        raise LinkError("bad vul: it must be none, ns, ew or both")
    seat = link_text(body, "seat", 5).upper()
    if seat and (seat not in SEAT_LETTERS or len(seat) != 1):
        raise LinkError("bad seat: it must be N, E, S or W")
    bots_text = link_text(body, "bots", 5).upper()
    if bots_text and (bots_text != "-" and any(b not in SEAT_LETTERS for b in bots_text)):
        raise LinkError("bad bots: the chairs the nets play, like EW, or - for none")
    if bots_text:                                          # any mix
        bots = {SEAT_LETTERS.index(b) for b in bots_text if b != "-"}
    elif seat:                                             # one chair, as on /
        bots = {0, 1, 2, 3} - {SEAT_LETTERS.index(seat)}
    else:                                                  # every chair is yours
        bots = set()
    if len(bots) == 4:
        raise LinkError("bad bots: keep at least one chair for yourself")
    seat = seat or "S"
    model, play_model = link_text(body, "model", 64), link_text(body, "play_model", 64)
    if model and model not in bid_models():
        raise LinkError(f"bidding model {model!r} is not on this server")
    if play_model and play_model not in play_models():
        raise LinkError(f"card-play model {play_model!r} is not on this server")

    hints = link_text(body, "hints", 5)
    if hints not in ("", "0", "1"):
        raise LinkError("bad hints: it must be 0 or 1")
    calls = parse_tokens(link_text(body, "auction", 600), "call", CALL_WORDS.get)
    cards = parse_tokens(link_text(body, "play", 300), "card", card_word)
    game = new_board(user_seat=SEAT_LETTERS.index(seat),
                     model=model or engine.default_bid_model(),
                     play_model=play_model or engine.default_play_model(),
                     owners=owners, board_no=prev["board_no"] + 1,
                     hints=prev["hints"] if hints == "" else hints == "1", peek=prev["peek"],
                     dealer=SEAT_LETTERS.index(dealer), vul=DEBUG_VULS[vul], debug=True, bots=bots)
    game["search"] = prev.get("search", engine.CONFIG.search)
    replay_moves(game, "".join(DESK.CALL_CHARS[c] for c in calls),
                 playdesk.encode_cards(cards))
    return game


def hand_text(owners, seat):
    """A whole hand as the debug link spells it, played cards included."""
    return ".".join("".join(RANKS[c % 13] for c in range(s * 13, s * 13 + 13) if owners[c] == seat)
                    or "-" for s in range(4))


def debug_code(game):
    """The readable link's pieces. `auction` and `play` are lists, one entry per call and
    per card, so the page can cut them back to the frame it is drawing. `bm` and `pm` are
    "" for the default models, which the link leaves out."""
    pg = game["pg"]
    return {
        **{key: hand_text(game["owners"], seat) for seat, key in enumerate("nesw")},
        "dealer": SEAT_LETTERS[game["dealer"]], "vul": DEBUG_VUL_CODES[game["vul"]],
        "auction": [call_token(c) for c in game["calls"]],
        "play": [card_name(c) for c in pg["played"]] if pg is not None else [],
        "seat": SEAT_LETTERS[game["view_seat"]],
        "bots": "".join(SEAT_LETTERS[b] for b in sorted(game["bots"] or ())),
        "bm": "" if game["model"] == engine.default_bid_model() else game["model"],
        "pm": "" if game["play_model"] == engine.default_play_model() else game["play_model"],
        "hints": int(bool(game["hints"])),
    }


# ---------------------------------------------------------------- challenge
#
# A short team match against the nets. The user sits South with three nets, and
# at the "other table" four nets play the same deals. Each board is compared in
# IMPs as it ends. The other table is played once, when the challenge starts, and
# kept: with search on a net may pick a different card each time, so replaying it
# would change the score under the user. The challenge link carries those results,
# so a friend who opens it plays against exactly the same other table.

CHALLENGE_BOARDS = 4
CHALLENGE_MAX = 8
CHALLENGE_SEAT = 2          # South


class Cancelled(Exception):
    pass


def robot_board(owners, model, play_model, search, tick=None):
    """The other table: all four chairs are nets. Returns the finished game dict.

    `tick(cards)` is called after every call (cards=0) and card; it returns False to stop.
    """
    def step(cards):
        if tick is not None and not tick(cards):
            raise Cancelled()
    game = new_board(user_seat=CHALLENGE_SEAT, model=model, play_model=play_model, owners=owners)
    game["search"] = search
    for _ in range(100):
        if auction_of(game).ended:
            break
        view, _ = bid_view(game)
        if view is None or not add_call(game, view["pick"]):
            raise RuntimeError("the bidding net could not bid at the other table")
        step(0)
    while phase(game) == "play":
        pg, batch = game["pg"], batch_for(game)
        card = playdesk.net_card(pg, playdesk.contracts_of(pg), batch)
        if card is None or not playdesk.play_card(pg, card, batch):
            raise RuntimeError("the card-play net could not play at the other table")
        game.pop("_batch", None)
        step(len(pg["played"]))
    return game


def board_summary(game):
    """One finished board, as the comparison shows it. N/S score, South's view."""
    r = result_view(game)
    ct = contract_view(game)
    return {
        "auction": [{"seat": (game["dealer"] + i) % 4, "name": call_name(c)}
                    for i, c in enumerate(game["calls"])],
        "contract": ct,
        "passed_out": r["passed_out"],
        "tricks": r["tricks"], "delta": r.get("delta"),
        "ns_score": r["ns_score"],
    }


def moves_code(game):
    pg = game["pg"]
    return ("".join(DESK.CALL_CHARS[c] for c in game["calls"]),
            playdesk.encode_cards(pg["played"]) if pg is not None else "")


def new_challenge(model, play_model, search, n=CHALLENGE_BOARDS, progress=None):
    """`progress(board index, cards played there)` returns False to cancel (raises `Cancelled`)."""
    rng = np.random.default_rng()
    deals, bots = [], []
    for k in range(n):
        owners = np.asarray(deal_owners(rng), dtype=np.int64)
        deals.append(owners)
        game = robot_board(owners, model, play_model, search,
                           tick=None if progress is None else (lambda cards, k=k: progress(k, cards)))
        bots.append(dict(board_summary(game), code=moves_code(game)))
    return {"deals": deals, "bot": bots, "mine": [None] * n, "i": 0,
            "model": model, "play_model": play_model, "search": bool(search),
            "player": None, "rated": [None] * n}


def challenge_from_link(body, prev):
    """A challenge from its link, or `LinkError`. The other table is replayed, not re-run."""
    def parts(key):
        v = body.get(key) or ""
        if not isinstance(v, str) or len(v) > 2000:
            raise LinkError(f"bad {key} in challenge link")
        return v.split(".")
    deals = parts("deals")
    auctions, played = parts("auctions"), parts("played")
    if not 1 <= len(deals) <= CHALLENGE_MAX or deals == [""]:
        raise LinkError(f"a challenge has 1 to {CHALLENGE_MAX} boards")
    if len(auctions) != len(deals) or len(played) != len(deals):
        raise LinkError("the challenge link needs one auction and one play per board")
    model = body.get("model") or prev["model"]
    play_model = body.get("play_model") or prev["play_model"]
    if model not in bid_models():
        raise LinkError(f"bidding model {model!r} is not on this server")
    if play_model not in play_models():
        raise LinkError(f"card-play model {play_model!r} is not on this server")
    search = str(body.get("search", "1")) in ("1", "true", "True")
    out, bots = [], []
    for k, code in enumerate(deals):
        owners = DESK.decode_deal(code) if code else None
        if owners is None:
            raise LinkError(f"board {k + 1} in challenge link has a bad deal")
        owners = np.asarray(owners, dtype=np.int64)
        game = new_board(user_seat=CHALLENGE_SEAT, model=model, play_model=play_model, owners=owners)
        try:
            replay_moves(game, auctions[k], played[k])
        except LinkError as e:
            raise LinkError(f"board {k + 1}, other table: {e}")
        if phase(game) != "over":
            raise LinkError(f"board {k + 1}, other table: the board is not finished")
        out.append(owners)
        bots.append(dict(board_summary(game), code=(auctions[k], played[k])))
    return {"deals": out, "bot": bots, "mine": [None] * len(out), "i": 0,
            "model": model, "play_model": play_model, "search": search,
            "player": None, "rated": [None] * len(out)}


def challenge_board(ch, prev_board_no):
    """A game dict for the challenge's current board, the user in South."""
    game = new_board(user_seat=CHALLENGE_SEAT, model=ch["model"], play_model=ch["play_model"],
                     hints=False, peek=False, owners=ch["deals"][ch["i"]],
                     board_no=prev_board_no + 1)
    game["search"] = ch["search"]
    game["challenge"] = ch
    if ch["player"]:
        players.board_started(ch["player"], DESK.encode_deal(ch["deals"][ch["i"]]))
    return game


def challenge_code(ch):
    """The pieces of the challenge link: every deal and the other table's moves."""
    return {
        "c": ".".join(DESK.encode_deal(o) for o in ch["deals"]),
        "ca": ".".join(b["code"][0] for b in ch["bot"]),
        "cp": ".".join(b["code"][1] for b in ch["bot"]),
        "m": ch["model"], "pm": ch["play_model"], "r": int(ch["search"]),
    }


def challenge_view(game):
    """What the page shows about the match. The other table's result for a board is
    sent only once the user's own board is over, so it cannot be peeked at."""
    ch = game.get("challenge")
    if ch is None:
        return None
    i = ch["i"]
    if ch["mine"][i] is None and phase(game) == "over":
        ch["mine"][i] = dict(board_summary(game), code=moves_code(game))
        if ch["player"]:
            ch["rated"][i] = players.board_finished(
                ch["player"], DESK.encode_deal(ch["deals"][i]),
                imps(ch["mine"][i]["ns_score"] - ch["bot"][i]["ns_score"]))
    boards, total = [], 0
    for k, bot in enumerate(ch["bot"]):
        mine = ch["mine"][k]
        row = {"no": k + 1, "status": "done" if mine else "playing" if k == i else "waiting",
               "mine": None, "bot": None, "imps": None, "rating": None}
        if mine:
            # The moves of both tables go out only once the board is over, for the review links.
            row["mine"] = {x: mine[x] for x in mine if x != "code"}
            row["bot"] = {x: bot[x] for x in bot if x != "code"}
            row["moves"] = {"mine": mine["code"], "bot": bot["code"]}
            row["imps"] = imps(mine["ns_score"] - bot["ns_score"])
            row["rating"] = ch["rated"][k]
            total += row["imps"]
        boards.append(row)
    return {"n": len(ch["bot"]), "i": i, "boards": boards, "total": total,
            "done": all(m is not None for m in ch["mine"]),
            "last": i == len(ch["bot"]) - 1,
            "rated": bool(ch["player"]),
            "code": challenge_code(ch)}


# A challenge takes seconds to deal: the bots play every board at the other table.
# The page asks for it as a job, outside the table lock, so the other tables keep
# moving, the page can show how far it is, and the user can cancel.

CH_JOBS = {"lock": threading.Lock(), "jobs": OrderedDict()}
CH_JOB_TTL = 600


def _challenge_worker(job):
    def progress(k, cards):
        job["board"], job["cards"] = k, cards
        return not job["cancel"]
    try:
        ch = new_challenge(job["model"], job["play_model"], job["search"], job["n"], progress)
        job.update(state="ready", ch=ch, board=job["n"], cards=0)
    except Cancelled:
        job["state"] = "cancelled"
    except Exception as exc:                        # keep the desk alive on any failure
        job.update(state="error", error=f"{type(exc).__name__}: {exc}")


def challenge_job_view(job):
    return {"job": job["id"], "state": job["state"], "n": job["n"], "board": job["board"],
            "cards": job["cards"], "error": job.get("error")}


# ------------------------------------------------------------------- routes

def body_of(request_):
    """The JSON body, or {} — never a 400.

    `request.json` raises when a client sends `Content-Type: application/json`
    with an empty body, and Flask answers with an HTML error page. No endpoint
    here has a required body, so a missing one should mean "no options", not a
    failure the caller has to parse out of HTML.
    """
    return request_.get_json(silent=True) or {}


def int_field(body, key):
    """An integer field of the JSON body, or -1 when it is missing or not a number."""
    try:
        return int(body.get(key, -1))
    except (TypeError, ValueError):
        return -1


def with_game(fn):
    def wrapped():
        gid = request.headers.get("X-Game", "")[:64] or "default"
        with LOCK:
            if gid not in GAMES:
                GAMES[gid] = new_board()
                advance(GAMES[gid])       # the nets bid up to the user's first turn
            GAMES.move_to_end(gid)
            while len(GAMES) > MAX_GAMES:
                GAMES.popitem(last=False)
            follow_turn(GAMES[gid])
            return fn(GAMES[gid])
    wrapped.__name__ = fn.__name__
    return wrapped


def register(app):
    players.register(app)

    @app.get("/")
    @app.get("/table")
    def table_page():
        return send_from_directory(STATIC_DIR, "table.html")

    @app.get("/debug")
    def debug_page():
        """The same page; it sees /debug in its own address and loads through /api/table/debug."""
        return send_from_directory(STATIC_DIR, "table.html")

    # The phone stylesheet and what "Add to Home Screen" reads. Named one by one, so
    # nothing else in the static folder is served by accident.
    @app.get("/<any('table.mobile.css', 'table.webmanifest', 'table-icon.svg', 'table-icon-180.png'):name>")
    def table_file(name):
        kind = "application/manifest+json" if name.endswith(".webmanifest") else None
        return send_from_directory(STATIC_DIR, name, mimetype=kind)

    @app.get("/api/table/state")
    @with_game
    def api_table_state(game):
        return jsonify(state_dump(game))

    @app.post("/api/table/new_board")
    @with_game
    def api_table_new_board(game):
        body = body_of(request)
        seat = int(body.get("seat", game["view_seat"])) % 4
        # A debug table keeps its dealer, vulnerability and the nets' chairs; / always goes
        # back to North, nobody.
        keep = ({"dealer": game["dealer"], "vul": game["vul"], "debug": True, "bots": game["bots"]}
                if game["debug"] else {})
        fresh = new_board(user_seat=seat, model=body.get("model") or game["model"],
                          play_model=body.get("play_model") or game["play_model"],
                          hints=game["hints"], peek=game["peek"],
                          board_no=game["board_no"] + 1, **keep)
        game.clear()
        game.update(fresh)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/load")
    @with_game
    def api_table_load(game):
        """Open a shared link in this tab's own game. Stops exactly at the link's position."""
        try:
            fresh = board_from_link(body_of(request), game)
        except LinkError as e:
            return jsonify(error=str(e)), 400
        game.clear()
        game.update(fresh)
        return jsonify(state_dump(game))

    @app.post("/api/table/debug")
    @with_game
    def api_table_debug(game):
        """Open the /debug page's readable link (no hands: a random deal). Stops at its position."""
        try:
            fresh = debug_board(body_of(request), game)
        except LinkError as e:
            return jsonify(error=str(e)), 400
        game.clear()
        game.update(fresh)
        return jsonify(state_dump(game))

    def no_debug(fn):
        """Challenges are rated; a debug table never starts one."""
        def wrapped(game):
            if game.get("debug"):
                return jsonify(error="no challenges on the debug table"), 400
            return fn(game)
        wrapped.__name__ = fn.__name__
        return wrapped

    @app.post("/api/table/challenge")
    @with_game
    @no_debug
    def api_table_challenge(game):
        """Start a challenge: deal the boards and play the other table now."""
        n = int_field(body_of(request), "boards")
        n = CHALLENGE_BOARDS if n < 1 else min(n, CHALLENGE_MAX)
        ch = new_challenge(game["model"], game["play_model"], game.get("search", engine.CONFIG.search), n)
        # Only a challenge the server dealt is rated; one from a link could be written by hand.
        ch["player"] = players.current_id()
        players.forfeit_open(ch["player"])       # boards left unfinished before count as lost
        fresh = challenge_board(ch, game["board_no"])
        game.clear()
        game.update(fresh)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/challenge/job")
    @with_game
    @no_debug
    def api_table_challenge_job(game):
        n = int_field(body_of(request), "boards")
        n = CHALLENGE_BOARDS if n < 1 else min(n, CHALLENGE_MAX)
        job = {"id": secrets.token_hex(8), "player": players.current_id(), "at": time.time(),
               "model": game["model"], "play_model": game["play_model"],
               "search": game.get("search", engine.CONFIG.search), "n": n,
               "state": "running", "board": 0, "cards": 0, "cancel": False}
        with CH_JOBS["lock"]:
            for key in [k for k, j in CH_JOBS["jobs"].items() if time.time() - j["at"] > CH_JOB_TTL]:
                CH_JOBS["jobs"][key]["cancel"] = True
                del CH_JOBS["jobs"][key]
            CH_JOBS["jobs"][job["id"]] = job
        threading.Thread(target=_challenge_worker, args=(job,), daemon=True).start()
        return jsonify(challenge_job_view(job))

    def own_job():
        with CH_JOBS["lock"]:
            job = CH_JOBS["jobs"].get(str(request.args.get("job") or body_of(request).get("job") or ""))
        return job if job is not None and job["player"] == players.current_id() else None

    @app.get("/api/table/challenge/progress")
    def api_table_challenge_progress():
        job = own_job()
        if job is None:
            return jsonify(error="no such challenge being dealt"), 404
        return jsonify(challenge_job_view(job))

    @app.post("/api/table/challenge/cancel")
    def api_table_challenge_cancel():
        job = own_job()
        if job is not None:
            job["cancel"] = True
            with CH_JOBS["lock"]:
                CH_JOBS["jobs"].pop(job["id"], None)
        return jsonify(ok=True)

    @app.post("/api/table/challenge/start")
    @with_game
    @no_debug
    def api_table_challenge_start(game):
        job = own_job()
        if job is None or job["state"] != "ready":
            return jsonify(error="that challenge is not dealt yet"), 400
        with CH_JOBS["lock"]:
            CH_JOBS["jobs"].pop(job["id"], None)
        ch = job["ch"]
        ch["player"] = job["player"]
        players.forfeit_open(ch["player"])       # boards left unfinished before count as lost
        fresh = challenge_board(ch, game["board_no"])
        game.clear()
        game.update(fresh)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/challenge/load")
    @with_game
    @no_debug
    def api_table_challenge_load(game):
        """Open a challenge link: the same deals and the same other table, from board 1."""
        try:
            ch = challenge_from_link(body_of(request), game)
        except LinkError as e:
            return jsonify(error=str(e)), 400
        fresh = challenge_board(ch, game["board_no"])
        game.clear()
        game.update(fresh)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/challenge/next")
    @with_game
    def api_table_challenge_next(game):
        ch = game.get("challenge")
        if ch is None:
            return jsonify(error="no challenge is running"), 400
        if phase(game) != "over":
            return jsonify(error="finish this board first"), 400
        challenge_view(game)                       # records this board if not yet
        if ch["i"] + 1 >= len(ch["deals"]):
            return jsonify(error="that was the last board"), 400
        ch["i"] += 1
        fresh = challenge_board(ch, game["board_no"])
        game.clear()
        game.update(fresh)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/restart")
    @with_game
    def api_table_restart(game):
        if game.get("challenge"):
            return jsonify(error="a challenge board cannot be replayed"), 400
        reset_board(game)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/hints")
    @with_game
    def api_table_hints(game):
        body = body_of(request)
        if game.get("challenge") and any(k in body for k in ("hints", "peek", "search")):
            return jsonify(error="hints, the solver and search are fixed during a challenge"), 400
        if "hints" in body:
            game["hints"] = bool(body["hints"])
        if "peek" in body:
            game["peek"] = bool(body["peek"])
        if "search" in body:
            game["search"] = bool(body["search"])
            if game["pg"] is not None:
                game["pg"]["search"] = game["search"]
        return jsonify(state_dump(game))

    @app.post("/api/table/models")
    @with_game
    def api_table_models(game):
        body = body_of(request)
        if game.get("challenge"):
            return jsonify(error="the models are fixed during a challenge"), 400
        if body.get("model") in bid_models():
            game["model"] = body["model"]
        if body.get("play_model") in play_models():
            game["play_model"] = body["play_model"]
            if game["pg"] is not None:
                game["pg"]["model"] = body["play_model"]
        return jsonify(state_dump(game))

    @app.post("/api/table/call")
    @with_game
    def api_table_call(game):
        if phase(game) != "auction":
            return jsonify(error="the auction is over"), 400
        if auction_of(game).turn not in user_chairs(game):
            return jsonify(error="it is not your turn to call"), 400
        call = int_field(body_of(request), "call")
        if not add_call(game, call):
            return jsonify(error=f"{call_name(call) if 0 <= call < N_CALLS else call} "
                                 f"is not a legal call here"), 400
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/card")
    @with_game
    def api_table_card(game):
        if phase(game) != "play":
            return jsonify(error="there is nothing to play"), 400
        pg, batch = game["pg"], batch_for(game)
        if not user_plays(game, int(batch.to_play()[0])):
            return jsonify(error="it is not your card to play"), 400
        card = int_field(body_of(request), "card")
        if not playdesk.play_card(pg, card, batch):
            return jsonify(error=f"{card_name(card) if 0 <= card < 52 else card} "
                                 f"is not a legal card here"), 400
        game.pop("_batch", None)
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/advance")
    @with_game
    def api_table_advance(game):
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/step")
    @with_game
    def api_table_step(game):
        """/debug: the nets make the one call or card on turn, whoever's chair it is."""
        if not game["debug"]:
            return jsonify(error="the nets step for you only on the debug table"), 400
        if not net_act(game):
            return jsonify(error="nothing for the nets to do here"), 400
        advance(game)
        return jsonify(state_dump(game))

    @app.post("/api/table/finish")
    @with_game
    def api_table_finish(game):
        """Let the nets run the rest of the deal out, for a user who is only watching.
        On /debug they take every chair from here to the end, the user's too."""
        if game["debug"]:
            for _ in range(60 + 52):
                if not net_act(game):
                    break
            return jsonify(state_dump(game))
        for _ in range(60):
            if user_on_turn(game) or phase(game) == "over":
                break
            if not advance(game):
                break
        return jsonify(state_dump(game))

    @app.post("/api/table/undo")
    @with_game
    def api_table_undo(game):
        if game.get("challenge"):
            return jsonify(error="a challenge board is played once, without undo"), 400
        if not undo(game):
            return jsonify(error="nothing of yours to take back yet"), 400
        advance(game)             # only moves when the user is not on turn, e.g. as dummy
        return jsonify(state_dump(game))

    @app.post("/api/table/claim")
    @with_game
    def api_table_claim(game):
        if not can_claim(game):
            if game.get("challenge"):
                return jsonify(error="a challenge board is played out, without claims"), 400
            return jsonify(error="you can claim when it is your card to play"), 400
        if not claim(game):
            return jsonify(error="Claim not accepted: the other side can still take a trick "
                                 "with the right defence. Play on."), 400
        return jsonify(state_dump(game))

    @app.post("/api/table/explain")
    @with_game
    def api_table_explain(game):
        """Queue the imagined-hands rollout for the call on turn; newest request wins."""
        if not game["hints"]:
            return jsonify(error="hints are off"), 400
        bot = bid_bot(game)
        if bot is None or not getattr(bot, "explains", False):
            return jsonify(error="this model cannot be rolled out"), 400
        if phase(game) != "auction" or auction_of(game).turn not in user_chairs(game):
            return jsonify(error="nothing to roll out here"), 400
        samples = max(32, min(256, int((body_of(request)).get("samples", 128))))
        view, legal = bid_view(game)
        ranked = sorted(((p, c) for c, p in enumerate(view.get("policy") or []) if legal[c]),
                        reverse=True)[:2]
        candidates = [c for _, c in ranked]
        key = (f"{game['model']}|{DESK.encode_deal(game['owners'])}|"
               f"{''.join(DESK.CALL_CHARS[c] for c in game['calls'])}|{samples}|"
               f"{game['dealer']}{playdesk.vul_code(*game['vul'])}")
        # `explain` deals from North. Turn the table so the dealer sits there (the nets see
        # seats relative to the dealer, so nothing else changes) and turn the seats back after.
        d = game["dealer"]
        snapshot = {"model": game["model"], "calls": list(game["calls"]),
                    "bitmaps": np.roll(game["bitmaps"], -d, axis=0), "dealer": d,
                    "vul": game["vul"] if d % 2 == 0 else game["vul"][::-1],
                    "candidates": candidates, "samples": samples}
        with EXPLAIN["lock"]:
            done = EXPLAIN["jobs"].get(key)
            if done and done["status"] == "done":
                return jsonify(job=key, **done)
            EXPLAIN["jobs"].setdefault(key, {"status": "queued"})
            EXPLAIN["next"] = (key, snapshot)
            if EXPLAIN["worker"] is None or not EXPLAIN["worker"].is_alive():
                EXPLAIN["worker"] = threading.Thread(target=_explain_worker, daemon=True)
                EXPLAIN["worker"].start()
        return jsonify(job=key, status="queued")

    @app.get("/api/table/explain")
    def api_table_explain_result():
        key = request.args.get("job", "")
        with EXPLAIN["lock"]:
            entry = EXPLAIN["jobs"].get(key)
        return jsonify(job=key, **(entry or {"status": "unknown"}))


def load(desk):
    """Take the live bid desk module; both model families come from the other two desks.

    `python -m emergent.bidserver` imports the bid desk a second time under a
    different name, so importing it from here would find an empty model registry.
    The caller hands over the object that actually loaded the checkpoints.
    """
    global DESK
    DESK = desk
    torch.set_num_threads(1)
