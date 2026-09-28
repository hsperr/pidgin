"""Machine APIs on the bid desk's Flask app, for other sites' tables to seat our bots.

- ``GET /apis/bbo.php`` — the BridgeBase robot (``u_bm/robot.php``) contract: same query
  (``pov d v n e s w h``, ``botstyle`` ignored) and the same ``<sc_bm><r type="bid" .../>``
  XML answer: ``<r type="bid" .../>`` during the auction, ``<r type="play" card=.../>``
  once ``h`` goes on with the cards played.
- ``GET /apis/brill/{,bid,lead,play}`` — Brill's Seat Robot API
  (https://brill.aalborgdata.dk/seat-api.html), BEN-compatible. Bidding and card play.

Both are stateless: each request carries the whole position and is answered from it
alone. ``model=<id>`` / ``play_model=<id>`` pick a checkpoint from models/models.json /
models/play_models.json; the default is each list's first entry. Only four-seat bidding
models are served here. Dealer and vulnerability are real inputs to the net (the desks
fix dealer North, nobody vulnerable; the APIs do not).

Card play needs the full 52-card layout for the play engine's bookkeeping, but a seat
only knows its own hand, dummy and the cards played. Who played each card is recovered
from the trick rules, and the unseen cards are dealt at random (seeded by the request, and
respecting suits a seat showed out of). The net's input for the seat on turn reads only its
own hand, the face-up partner hand, and the played cards, so the filler never reaches it:
tests/test_apis.py checks that the chosen card does not change when the filler does.
"""

from __future__ import annotations

import hashlib
import threading
from xml.sax.saxutils import quoteattr

import numpy as np
import torch
from flask import Response, jsonify, request

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import DOUBLE, PASS, REDOUBLE
from bridgezero.fourseat.model import competitive_log_probs, policy_log_probs
from bridgezero.fourseat.state import features_from_history
from emergent import playdesk
from emergent.deck import RANKS, SUITS, call_token, card_name, trick_best

SEARCH_LOCK = threading.Lock()

DESK = None                      # the live bid desk module, set by load()
SEATS = "NESW"
STRAINS = "CDHSN"                # bid index = (level - 1) * 5 + strain (cards are S H D C)


class ApiError(ValueError):
    """A malformed request: answered with HTTP 400 and {"error": ...}, never a guessed call."""


def load(desk):
    global DESK
    DESK = desk


# ------------------------------------------------------------------ parsing

def parse_hand(text: str, name: str = "hand") -> list[int]:
    """``AKQ2.J54.T98.762`` (S.H.D.C; ``_`` accepted for ``.``) -> 13 card indices."""
    parts = (text or "").replace("_", ".").upper().split(".")
    if len(parts) != 4:
        raise ApiError(f"'{name}' must be four suits S.H.D.C separated by '.', got {text!r}")
    cards = []
    for s, run in enumerate(parts):
        for ch in run.replace("10", "T"):
            if ch not in RANKS:
                raise ApiError(f"'{name}' has an unknown rank {ch!r}")
            cards.append(s * 13 + RANKS.index(ch))
    if len(cards) != 13 or len(set(cards)) != 13:
        raise ApiError(f"'{name}' must hold 13 different cards, got {len(set(cards))}")
    return cards


def parse_card(token: str) -> int:
    t = token.upper()
    if len(t) != 2 or t[0] not in SUITS or t[1] not in RANKS:
        raise ApiError(f"bad card {token!r}: want suit S/H/D/C then rank AKQJT98765432")
    return SUITS.index(t[0]) * 13 + RANKS.index(t[1])


def parse_call(token: str) -> int:
    """Any common spelling of a call -> call index (0..34 bids, 35 P, 36 X, 37 XX)."""
    t = token.strip().upper()
    if t in ("--", "P", "PASS"):
        return PASS
    if t in ("DB", "X", "D", "DBL", "DOUBLE"):
        return DOUBLE
    if t in ("RD", "XX", "R", "RDBL", "REDOUBLE"):
        return REDOUBLE
    if t.endswith("NT"):
        t = t[:-1]
    if len(t) == 2 and t[0] in "1234567" and t[1] in STRAINS:
        return (int(t[0]) - 1) * 5 + STRAINS.index(t[1])
    raise ApiError(f"bad call {token!r}")


def parse_seat(text: str, name: str) -> int:
    t = (text or "").strip().upper()[:1]
    if t not in SEATS or not t:
        raise ApiError(f"'{name}' must be N, E, S or W")
    return SEATS.index(t)


def parse_vul(text: str) -> tuple[bool, bool]:
    t = (text or "").strip().upper()
    table = {"": (False, False), "NONE": (False, False), "-": (False, False), "O": (False, False),
             "NS": (True, False), "N": (True, False), "EW": (False, True), "E": (False, True),
             "ALL": (True, True), "BOTH": (True, True), "B": (True, True)}
    if t not in table:
        raise ApiError(f"bad vulnerability {text!r}: want None, NS, EW or All")
    return table[t]


def brill_calls(ctx: str) -> list[int]:
    """Brill ``ctx``: two-character tokens with no separator (``1S--2H``); the optional
    dash-separated form (``1S-P-2H``) is accepted too."""
    ctx = (ctx or "").strip()
    if len(ctx) % 2 == 0:
        try:
            return [parse_call(ctx[i:i + 2]) for i in range(0, len(ctx), 2)]
        except ApiError:
            pass
    return [parse_call(t) for t in ctx.split("-") if t]


def bid_token(call: int, dialect: str) -> str:
    if call == PASS:
        return "--" if dialect == "brill" else "P"
    if call == DOUBLE:
        return "Db" if dialect == "brill" else "X"
    if call == REDOUBLE:
        return "Rd" if dialect == "brill" else "XX"
    return f"{call // 5 + 1}{STRAINS[call % 5]}"


def check_auction(calls: list[int]) -> AuctionState:
    st = AuctionState.from_calls([])
    for i, c in enumerate(calls):
        if st.ended or not st.legal_mask()[c]:
            raise ApiError(f"call {i + 1} ({bid_token(c, 'bbo')}) is not legal in this auction")
        st = AuctionState.from_calls(calls[:i + 1])
    return st


# ------------------------------------------------------------------ bidding

def bid_model(model_id: str | None):
    models = DESK.MODELS
    mid = model_id or next(iter(models))
    bot = models.get(mid)
    if bot is None:
        raise ApiError(f"unknown model {mid!r}; known: {', '.join(models)}")
    if bot.family != "fourseat":
        raise ApiError(f"model {mid!r} is not a four-seat bidding net")
    return mid, bot


def auction_features(bot, calls: list[int], dealer: int, vul_ns: bool, vul_ew: bool) -> torch.Tensor:
    """The net's auction input with the real dealer and vulnerability."""
    hist = torch.tensor([calls], dtype=torch.long) if calls else torch.zeros(1, 0, dtype=torch.long)
    actor = torch.tensor([(dealer + len(calls)) % 4])
    feats = features_from_history(hist, torch.tensor([dealer]), torch.tensor([float(vul_ns)]),
                                  torch.tensor([float(vul_ew)]), actor, doubles=bot.doubles)
    if not bot.competitive:
        return feats
    extra = torch.zeros(1, 2)
    if calls:
        pos = torch.arange(hist.shape[1])[None]
        none = torch.full_like(hist, -1)
        last_bid = torch.where(hist < PASS, pos, none).max(1).values
        last_active = torch.where(hist != PASS, pos, none).max(1).values
        last_xx = torch.where(hist == REDOUBLE, pos, none).max(1).values
        extra[:, 0] = ((last_bid >= 0) & (len(calls) - 1 - last_active == 2)).float()
        extra[:, 1] = (last_xx > last_bid).float()
    return torch.cat((feats, extra), 1)


@torch.no_grad()
def choose_call(bot, hand: list[int], calls: list[int], dealer: int, vul: tuple[bool, bool]):
    """(call, [(call, prob), ...] best first) — greedy, as in every match."""
    st = check_auction(calls)
    if st.ended:
        raise ApiError("the auction is already over")
    bitmap = torch.zeros(1, 52)
    bitmap[0, hand] = 1.0
    out = bot.net(bitmap, auction_features(bot, calls, dealer, *vul))
    n = bot.net.n_actions
    mask = torch.tensor([bot.mask(st)[:n].tolist()])
    probs = (competitive_log_probs(out, mask) if bot.competitive
             else policy_log_probs(out, mask))[0].exp()
    order = probs.argsort(descending=True).tolist()
    top = [(c, float(probs[c])) for c in order if mask[0, c]][:4]
    return top[0][0], top


def meaning(model_id: str, calls: list[int], call: int) -> tuple[str, bool]:
    """Plain-English meaning of ``call`` after ``calls``, from the model's self-play corpus.

    The corpus was built with dealer North and nobody vulnerable, keyed by the calls from
    the dealer, so it describes the same position in the model's own self-play. Returns
    (text, alert): alert when the named suit is usually not held 4+ long (artificial).
    """
    table = DESK.CORPUS.get(model_id) or {}
    entry = (table.get("-".join(call_token(c) for c in calls)) or {}).get("calls", {}).get(call_token(call))
    if call == PASS:
        return ("", False) if not entry else (f"{entry['hcp'][0]:.0f}-{entry['hcp'][2]:.0f} HCP", False)
    if not entry:
        if call in (DOUBLE, REDOUBLE):
            return ("penalty" if call == DOUBLE else "to play", False)
        return ("natural", False)
    lo, hi = entry["hcp"][0], entry["hcp"][2]
    parts = [f"{lo:.0f}-{hi:.0f} HCP"]
    alert = False
    if entry.get("suit"):
        four, five, six = entry["suit_len"]
        suit = {"S": "spades", "H": "hearts", "D": "diamonds", "C": "clubs"}[entry["suit"]]
        if six >= 0.8:
            parts.insert(0, f"6+ {suit}")
        elif five >= 0.8:
            parts.insert(0, f"5+ {suit}")
        elif four >= 0.8:
            parts.insert(0, f"4+ {suit}")
        elif four >= 0.5:
            parts.insert(0, f"usually 4+ {suit}")
        else:
            parts.insert(0, f"artificial, {suit} not promised")
            alert = True
    elif call % 5 == 4 and call < PASS:
        if entry.get("bal", 1) < 0.5:
            parts.append("strong, any shape")
            alert = True                       # not the standard balanced notrump
        else:
            parts.append("balanced")
    if call == DOUBLE:
        parts.insert(0, "penalty/values")
    return ", ".join(parts), alert


# ------------------------------------------------------------------ card play

def play_model(model_id: str | None):
    models = playdesk.MODELS
    if not models:
        raise ApiError("no card-play model is loaded")
    mid = model_id or next(iter(models))
    if mid not in models:
        raise ApiError(f"unknown play model {mid!r}; known: {', '.join(models)}")
    return mid, models[mid]


def trick_seats(played: list[int], leader: int, trump: int) -> tuple[list[int], int]:
    """Seat that played each card (from the trick rules), and the seat on turn now.
    ``trump`` in card order S H D C, 4 = notrump."""
    seats, turn = [], leader
    for i, card in enumerate(played):
        seats.append(turn)
        if i % 4 == 3:
            turn = (turn + 1 + trick_best(played[i - 3:i + 1], trump)) % 4
        else:
            turn = (turn + 1) % 4
    return seats, turn


def fill_owners(known: dict[int, int], need: dict[int, int], voids: dict[int, set],
                seed: int) -> np.ndarray:
    """52 owners: ``known`` card->seat, the rest dealt to hidden seats (``need`` counts),
    avoiding suits a seat showed out of. Only the play engine's bookkeeping reads these."""
    free = [c for c in range(52) if c not in known]
    rng = np.random.default_rng(seed)
    for attempt in range(200):
        order = rng.permutation(free).tolist()
        owners = np.full(52, -1, dtype=np.int64)
        for c, s in known.items():
            owners[c] = s
        left = dict(need)
        ok = True
        # most constrained seats first
        for card in sorted(order, key=lambda c: -sum(c // 13 in voids.get(s, ()) for s in left)):
            options = [s for s, k in left.items() if k and (attempt >= 100 or card // 13 not in voids.get(s, ()))]
            if not options:
                ok = False
                break
            s = options[int(rng.integers(len(options)))]
            owners[card] = s
            left[s] -= 1
        if ok:
            return owners
    raise ApiError("could not place the unseen cards consistently with the play so far")


@torch.no_grad()
def choose_card(bot, *, seat: int, hand: list[int], dummy: list[int] | None, played: list[int],
                calls: list[int], dealer: int, vul: tuple[bool, bool], seed_text: str,
                search: bool = True):
    """(card, [(card, prob), ...]) for the card this seat owes (dummy's when declarer asks).

    The card comes from the same PIMC search /table plays with (declarer always, defence
    from trick SEARCH_DEFENCE_FROM); the probabilities are the plain net's, for context.
    The search never reads the filled-in hidden hands: it samples its own layouts from
    what this seat can see."""
    check_auction(calls)
    contract = playdesk.contract_from_calls(calls, dealer)
    if contract is None:
        raise ApiError("the auction was passed out: there is no play")
    declarer = contract["declarer"]
    dummy_seat = (declarer + 2) % 4
    leader = (declarer + 1) % 4
    if len(set(played)) != len(played):
        raise ApiError("'played' repeats a card")
    if len(played) >= 52:
        raise ApiError("all 52 cards are already played")
    seats, turn = trick_seats(played, leader, int(contract["trump"]))
    if turn != seat and not (seat == declarer and turn == dummy_seat):
        raise ApiError(f"it is {SEATS[turn]}'s card, not {SEATS[seat]}'s "
                       f"(declarer {SEATS[declarer]}, {len(played)} cards played)")
    if seat == dummy_seat:
        raise ApiError("dummy is never asked: declarer plays dummy's cards")
    if played and dummy is None:
        raise ApiError("'dummy' is required once the opening lead is made")

    known = {c: seat for c in hand}
    if dummy is not None and played:
        for c in dummy:
            if c in known:
                raise ApiError("'hand' and 'dummy' share a card")
            known[c] = dummy_seat
    voids: dict[int, set] = {}
    for i, (card, who) in enumerate(zip(played, seats)):
        if card in known and known[card] != who:
            raise ApiError(f"{card_name(card)} was played by {SEATS[who]} but belongs to "
                           f"{SEATS[known[card]]}")
        known[card] = who
        start = i - i % 4
        if i % 4 and card // 13 != played[start] // 13:
            voids.setdefault(who, set()).add(played[start] // 13)
    count = {s: sum(1 for w in known.values() if w == s) for s in range(4)}
    if any(k > 13 for k in count.values()):
        raise ApiError("a seat holds more than 13 cards")
    need = {s: 13 - count[s] for s in range(4) if 13 - count[s] > 0}
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    owners = fill_owners(known, need, voids, seed)

    game = {"owners": owners, "calls": list(calls), "dealer": dealer, "synthetic": False,
            "trump": int(contract["trump"]), "declarer": declarer, "level": int(contract["level"]),
            "doubled": int(contract["doubled"]), "vul_ns": vul[0], "vul_ew": vul[1],
            "played": list(played), "bench_dd": None}
    c = playdesk.contracts_of(game)
    batch = playdesk.batch_of(game, c)
    if int(batch.to_play()[0]) != turn:
        raise ApiError("internal: play engine disagrees about whose turn it is")
    legal = batch.legal()[0]
    view = bot.view(c, batch, len(played))
    probs = torch.tensor(view["probs"])
    order = [int(i) for i in probs.argsort(descending=True) if legal[int(i)]]
    card = order[0]
    if search:
        with SEARCH_LOCK:                     # the searcher keeps the auction between calls
            player = bot.searcher()
            player.start(c)
            card = int(player.choose(batch, c, batch.legal(), len(played))[0])
    return card, [(i, float(probs[i])) for i in order[:4]]


# ------------------------------------------------------------------ routes

def error(message: str, code: int = 400):
    return jsonify(error=message), code


def register(app):
    @app.get("/apis/bbo.php")
    def api_bbo():
        """robot.php: ``h`` is the auction, then (once it is over) the cards played, all joined
        by ``-`` (``...-4h-p-p-p-H9``). A call is answered ``<r type="bid" .../>``, a card
        ``<r type="play" card="HJ"/>``. As at a table, only what ``pov`` can see is read: its
        own hand, dummy once the lead is down, and the history. When dummy is on turn the
        declarer is asked (``pov`` = declarer); ``pov`` = dummy is answered the same way."""
        a = request.args
        try:
            pov = parse_seat(a.get("pov", ""), "pov")
            dealer = parse_seat(a.get("d", ""), "d")
            vul = parse_vul(a.get("v", "-"))
            tokens = [t for t in a.get("h", "").strip().split("-") if t]
            calls, played = [], []
            for t in tokens:
                is_card = len(t) == 2 and t[0].upper() in SUITS and t[1].upper() in RANKS
                if is_card:
                    played.append(parse_card(t))
                elif played:
                    raise ApiError(f"call {t!r} after the play began")
                else:
                    calls.append(parse_call(t))
            st = check_auction(calls)
            if played and not st.ended:
                raise ApiError("cards in 'h' before the auction is over")
            if not st.ended:
                if (dealer + len(calls)) % 4 != pov:
                    raise ApiError(f"it is {SEATS[(dealer + len(calls)) % 4]}'s call, not {SEATS[pov]}'s")
                hand = parse_hand(a.get("nesw"[pov], ""), "nesw"[pov])
                mid, bot = bid_model(a.get("model"))
                call, _ = choose_call(bot, hand, calls, dealer, vul)
                text, _alert = meaning(mid, calls, call)
                answer = f'<r type="bid" bid="{bid_token(call, "bbo")}" meaning={quoteattr(text)}/>'
            else:
                contract = playdesk.contract_from_calls(calls, dealer)
                if contract is None:
                    raise ApiError("the auction was passed out: there is no play")
                declarer = contract["declarer"]
                dummy_seat = (declarer + 2) % 4
                seat = declarer if pov == dummy_seat else pov
                hand = parse_hand(a.get("nesw"[seat], ""), "nesw"[seat])
                dummy = None
                if played:
                    other = declarer if seat == dummy_seat else dummy_seat
                    dummy = parse_hand(a.get("nesw"[other], ""), "nesw"[other])
                mid, bot = play_model(a.get("play_model"))
                seed_text = "|".join(a.get(k, "") for k in ("d", "v", "pov", "n", "e", "s", "w", "h"))
                card, _ = choose_card(bot, seat=seat, hand=hand, dummy=dummy, played=played,
                                      calls=calls, dealer=dealer, vul=vul, seed_text=seed_text)
                answer = f'<r type="play" card="{card_name(card)}"/>'
        except ApiError as exc:
            body = f'<?xml version="1.0" encoding="UTF-8"?>\n<sc_bm error={quoteattr(str(exc))}/>\n'
            return Response(body, status=400, mimetype="text/xml")
        attrs = " ".join(f"{k}={quoteattr(a.get(k, ''))}" for k in ("pov", "d", "v", "n", "s", "e", "w", "h"))
        body = (f'<?xml version="1.0" encoding="UTF-8"?>\n<sc_bm {attrs} o="" bm="n" ac="n" rc="0" '
                f'c="n" model={quoteattr(mid)}>\n  {answer}\n</sc_bm>\n')
        return Response(body, mimetype="text/xml")

    @app.get("/apis/brill")
    @app.get("/apis/brill/")
    def api_brill_root():
        return jsonify(ok=True, api="Brill Seat Robot API (BEN-compatible)",
                       bid_models=list(DESK.MODELS), play_models=list(playdesk.MODELS),
                       endpoints=["/apis/brill/bid", "/apis/brill/lead", "/apis/brill/play"])

    def brill_common(a):
        seat = parse_seat(a.get("seat", ""), "seat")
        dealer = parse_seat(a.get("dealer", ""), "dealer")
        if "hand" not in a:
            raise ApiError("Missing 'hand' parameter")
        if "ctx" not in a:
            raise ApiError("Missing 'ctx' parameter")
        hand = parse_hand(a.get("hand", ""))
        vul = parse_vul(a.get("vul", ""))
        calls = brill_calls(a.get("ctx", ""))
        return seat, dealer, hand, vul, calls

    @app.get("/apis/brill/bid")
    def api_brill_bid():
        try:
            seat, dealer, hand, vul, calls = brill_common(request.args)
            if (dealer + len(calls)) % 4 != seat:
                raise ApiError(f"seat {SEATS[seat]} disagrees with dealer + ctx "
                               f"({SEATS[(dealer + len(calls)) % 4]} is to call)")
            mid, bot = bid_model(request.args.get("model"))
            call, top = choose_call(bot, hand, calls, dealer, vul)
            text, alert = meaning(mid, calls, call)
        except ApiError as exc:
            return error(str(exc))
        return jsonify(bid=bid_token(call, "brill"), alert=alert, explanation=text,
                       candidates=[{"bid": bid_token(c, "brill"), "score": round(p, 4)} for c, p in top],
                       model=mid)

    def brill_card(lead: bool):
        a = request.args
        try:
            seat, dealer, hand, vul, calls = brill_common(a)
            played_text = "" if lead else a.get("played", "")
            if len(played_text) % 2:
                raise ApiError("'played' must be two-character cards (SA, H7, ...)")
            played = [parse_card(played_text[i:i + 2]) for i in range(0, len(played_text), 2)]
            if lead and a.get("played"):
                raise ApiError("/lead is for the opening lead; use /play once a card is down")
            if not lead and not played:
                raise ApiError("/play needs 'played'; the opening lead is /lead")
            dummy = parse_hand(a["dummy"], "dummy") if a.get("dummy") else None
            mid, bot = play_model(a.get("play_model"))
            seed_text = "|".join(a.get(k, "") for k in ("board", "dealer", "vul", "seat", "hand", "dummy", "ctx", "played"))
            card, top = choose_card(bot, seat=seat, hand=hand, dummy=dummy, played=played,
                                    calls=calls, dealer=dealer, vul=vul, seed_text=seed_text)
        except ApiError as exc:
            return error(str(exc))
        return jsonify(card=card_name(card), model=mid,
                       candidates=[{"card": card_name(c), "score": round(p, 4)} for c, p in top])

    @app.get("/apis/brill/lead")
    def api_brill_lead():
        return brill_card(lead=True)

    @app.get("/apis/brill/play")
    def api_brill_play():
        return brill_card(lead=False)
