"""Play desk: watch the card-play net play out a contract.

Routes live on the bid desk's Flask app (`register(app)` from bidserver), so one
process serves both pages:

    GET  /play                  the page
    GET  /api/play/state        everything the page draws
    POST /api/play/setup        deal / contract / vulnerability, or a shared link
    POST /api/play/card         play one named card (the user)
    POST /api/play/net_card     play the net's card
    POST /api/play/auto         net plays on while the seat on turn is set to "net"
    POST /api/play/undo         take back the last card
    POST /api/play/restart      same board, trick one again
    POST /api/play/model        pick the checkpoint

Everything the net needs goes through the exact code it was trained with
(`bridgezero/play`, `bridgezero/bridge/play.py`, copied by sync_models.sh). The
game holds only the deal, the contract and the cards played so far; the
`PlayBatch` is rebuilt from that on every request, which makes undo free.

The double dummy numbers come from the same solver the benchmark's `dd_tricks`
came from (endplay/DDS): `calc_all_tables` for the board, and `solve_board` at
the live position for "what each card is worth". 52 solves cost ~50 ms, so the
per-card comparison is free. When the solver is missing, every DD field is None
and the page says so rather than guessing.
"""
import json
import os
import threading
from collections import OrderedDict

import numpy as np
import torch
from flask import jsonify, request, send_from_directory

from bridgezero.bridge.calls import CONTRACTS, DOUBLE, PASS, REDOUBLE
from bridgezero.bridge.deals import deal_to_pbn
from bridgezero.bridge.play import PlayBatch
from bridgezero.bridge.scoring import contract_score
from bridgezero.play.data import N_CALLS, Contracts, load_contracts
from bridgezero.play.model import PLAY_FEATURES, PlayNet, encode
from emergent.deck import deal_owners

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "bidserver_static")
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")
BENCH_FILE = os.path.join(MODELS_DIR, "bench_100k.npz")
BENCH_LIMIT = int(os.environ.get("PLAY_BENCH_LIMIT", "4000"))
# PIMC at the table. 20 layouts is where the offline gain flattens (+0.99 IMP a
# board over the plain net); the budget is what keeps the opening lead bearable on
# a one-core box, where a trick-one solve costs ~290 ms against a laptop's 6 ms.
SEARCH_SAMPLES = int(os.environ.get("PLAY_SEARCH_SAMPLES", "20"))
SEARCH_BUDGET_MS = float(os.environ.get("PLAY_SEARCH_BUDGET_MS", "900"))
# Defence searches too, but not before trick 2. Measured on 600 boards, starting at
# trick 2 gives the same defence regret as searching every turn (0.537) for 45% of the
# cost -- tricks 0 and 1 add nothing, and on this box they are nearly all of the price.
SEARCH_DEFENCE = os.environ.get("PLAY_SEARCH_DEFENCE", "all")
SEARCH_DEFENCE_FROM = int(os.environ.get("PLAY_SEARCH_DEFENCE_FROM", "2"))

RANKS = "AKQJT98765432"          # rank 0 = ace, 12 = two
SUITS = "SHDC"                   # card index = suit * 13 + rank
STRAINS = ["S", "H", "D", "C", "NT"]
SEAT_NAMES = ["North", "East", "South", "West"]
NAMES = [c[0] for c in CONTRACTS]
TRUMP_TO_BID_STRAIN = (3, 2, 1, 0, 4)     # cards are S,H,D,C,NT; bids are C,D,H,S,NT
BID_STRAIN_TO_TRUMP = (3, 2, 1, 0, 4)     # its own inverse
HCP_W = np.array([4, 3, 2, 1] + [0] * 9)
CARD_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"   # 52, one per card

MODELS = OrderedDict()    # id -> PlayBot
GAMES = OrderedDict()     # game id -> game dict, separate from the bid desk's
MAX_GAMES = 300
LOCK = threading.Lock()
BENCH = {"contracts": None, "tried": False}

# endplay is optional: without it the page still plays, it just has no DD answer.
try:
    from endplay.dds import calc_all_tables, solve_board
    from endplay.types import Card, Deal, Denom, Player, Rank
    DENOMS = (Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs, Denom.nt)
    PLAYERS = (Player.north, Player.east, Player.south, Player.west)
    DDS_RANKS = (Rank.RA, Rank.RK, Rank.RQ, Rank.RJ, Rank.RT, Rank.R9, Rank.R8,
                 Rank.R7, Rank.R6, Rank.R5, Rank.R4, Rank.R3, Rank.R2)
    SOLVER = None
except Exception as exc:                                   # pragma: no cover
    SOLVER = f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------- cards

def card_name(card):
    return SUITS[card // 13] + RANKS[card % 13]


def card_parts(card):
    return {"card": int(card), "suit": SUITS[card // 13], "rank": RANKS[card % 13]}


def encode_cards(cards):
    return "".join(CARD_CHARS[c] for c in cards)


def decode_cards(code):
    out = []
    for ch in str(code)[:52]:
        i = CARD_CHARS.find(ch)
        if i < 0:
            break
        out.append(i)
    return out


def hand_info(owners, seat, unplayed=None):
    held = [c for c in range(52) if owners[c] == seat and (unplayed is None or unplayed[c])]
    all_cards = [c for c in range(52) if owners[c] == seat]
    bits = np.zeros(52, dtype=np.int64)
    bits[all_cards] = 1
    sh = bits.reshape(4, 13)
    return {
        "seat": seat, "name": SEAT_NAMES[seat], "hcp": int((sh * HCP_W).sum()),
        "shape": [int(x) for x in sh.sum(1)],
        "suits": {SUITS[s]: [card_parts(c) for c in held if c // 13 == s] for s in range(4)},
        "spent": {SUITS[s]: [card_parts(c) for c in all_cards
                             if c // 13 == s and c not in held] for s in range(4)},
        "left": len(held),
    }


# ------------------------------------------------------------------- models

class PlayBot:
    """One card-play checkpoint, read through the classes it was trained with."""

    def __init__(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net = PlayNet(**state["config"])
        self.net.load_state_dict(state["net"])
        self.net.eval()
        self.step = state.get("step")
        cfg = state["config"]
        self.info = (f"PlayNet {cfg['width']}x{cfg['depth']} · {PLAY_FEATURES} inputs · "
                     f"policy over 52 cards + belief head")
        self._searcher = None

    def searcher(self):
        """PIMC over this bot's own net, built on first use.

        Built late because the solver's tables cost ~120 MB and most requests never
        search. The budget matters more than the sample count here: a solve is ~300x
        dearer at trick one than at trick seven, so a fixed count would stall the
        opening and waste effort at the end.
        """
        if self._searcher is None:
            from bridgezero.play.search import PIMCPlayer
            self._searcher = PIMCPlayer.from_net(self.net, SEARCH_SAMPLES,
                                                 budget_ms=SEARCH_BUDGET_MS,
                                                 defence=SEARCH_DEFENCE,
                                                 defence_from_trick=SEARCH_DEFENCE_FROM)
        return self._searcher

    @torch.no_grad()
    def view(self, contracts, batch, step):
        """Everything the net computes for the seat on turn, ready for the page."""
        auction = self.net.auction(contracts.calls, contracts.n_calls, contracts.dealer)
        enc = encode(batch, contracts, step > 0, auction)
        legal = batch.legal()
        out = self.net(enc["features"], legal)
        probs = out["log_probs"][0].exp()
        belief = torch.softmax(out["belief"][0], -1)        # (52, 4) over relative seats
        turn = int(batch.to_play()[0])
        pick = int(probs.argmax())
        feats = enc["features"][0].tolist()
        return {
            "seat": turn, "legal": legal[0].tolist(), "probs": probs.tolist(),
            "pick": pick, "pick_name": card_name(pick), "confidence": float(probs[pick]),
            "top": [{"card": int(c), "name": card_name(int(c)), "p": float(probs[c])}
                    for c in probs.argsort(descending=True)[:5] if legal[0, int(c)]],
            "belief": belief_view(belief, enc, batch, contracts, turn, step > 0),
            "input": input_view(feats),
        }


def visible_seats(contracts, turn, dummy_visible):
    """The hands the seat on turn may look at: its own, and the one that is face up.

    The declaring side is one player with two hands, so declarer sees dummy and dummy
    sees declarer; a defender sees dummy. Nothing is face up before the opening lead.
    """
    declarer, dummy = int(contracts.declarer[0]), (int(contracts.declarer[0]) + 2) % 4
    face_up = declarer if turn == dummy else dummy
    return {turn} | ({face_up} if dummy_visible else set())


def belief_view(belief, enc, batch, contracts, turn, dummy_visible):
    """Per hidden card: where the net thinks it sits, and where it really is.

    Two hands are hidden from the seat on turn, three at the opening lead before dummy
    shows. The head still scores all four relative seats, so the probabilities are
    renormalised over the seats that are actually possible and the mass the net spent on
    impossible ones is reported as `leak`.
    """
    hidden = enc["belief_mask"][0]
    owner = batch.owner[0]
    seen = visible_seats(contracts, turn, dummy_visible)
    candidates = [rel for rel in (1, 2, 3) if (turn + rel) % 4 not in seen]
    cards, right = [], 0
    for card in range(52):
        if not bool(hidden[card]):
            continue
        row = belief[card]
        mass = float(sum(row[r] for r in candidates)) or 1e-9
        true_seat = int(owner[card])
        probs = {str((turn + r) % 4): float(row[r]) / mass for r in candidates}
        best = max(probs, key=probs.get)
        right += int(best == str(true_seat))
        cards.append({"card": card, "name": card_name(card), "p": probs,
                      "true": true_seat, "leak": round(1.0 - mass, 4)})
    return {
        "cards": cards, "hidden": len(cards),
        "seats": [(turn + r) % 4 for r in candidates],
        "accuracy": (right / len(cards)) if cards else None,
    }


INPUT_LAYOUT = [
    ("my hand", 52), ("the face-up hand", 52), ("dummy is visible", 1),
    ("cards I played", 52), ("cards LHO played", 52),
    ("cards partner played", 52), ("cards RHO played", 52),
    ("this trick: card just before me", 52), ("this trick: two before me", 52),
    ("this trick: three before me", 52),
    ("trump (S H D C NT)", 5), ("level 1-7", 7), ("undoubled / X / XX", 3),
    ("my role (declarer, dummy, defender)", 3), ("vulnerable (us, them)", 2),
    ("tricks won (us, them)", 2), ("trick number / 13", 1),
    ("auction bag: my calls", N_CALLS), ("auction bag: LHO's calls", N_CALLS),
    ("auction bag: partner's calls", N_CALLS), ("auction bag: RHO's calls", N_CALLS),
    ("the auction read as a sequence (GRU)", 96),
]


def input_view(f):
    segments, i = [], 0
    for name, n in INPUT_LAYOUT:
        segments.append({"name": name, "values": f[i:i + n]})
        i += n
    assert i == len(f) == PLAY_FEATURES, (i, len(f))
    scalars = [
        ("dummy is face up", f[104]),
        ("I am declarer", f[484]), ("I am dummy", f[485]), ("I am a defender", f[486]),
        ("we are vulnerable", f[487]), ("they are vulnerable", f[488]),
        ("our tricks / 13", f[489]), ("their tricks / 13", f[490]),
        ("trick number / 13", f[491]),
    ]
    note = ("Every input but the last 96 is a plain 0/1 bit or a small fraction. Seats are "
            "relative: slot 0 is always the player on turn, then LHO, partner, RHO, so one "
            "net covers all four chairs. The last 96 numbers are the GRU's reading of the "
            "auction from this chair — learned, not readable.")
    return {"size": len(f), "segments": segments, "scalars": scalars, "note": note}


def load_models(models_dir=MODELS_DIR):
    path = os.path.join(models_dir, "play_models.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        manifest = json.load(fh)
    for m in manifest:
        file = os.path.join(models_dir, m["file"])
        if not os.path.exists(file):
            continue
        bot = PlayBot(file)
        bot.id, bot.label, bot.file = m["id"], m["label"], m["file"]
        bot.note = m.get("note", "")
        MODELS[bot.id] = bot


def models_list():
    return [{"id": b.id, "label": b.label, "step": b.step, "info": b.info, "note": b.note}
            for b in MODELS.values()]


# ------------------------------------------------------- contracts and games

def contract_from_calls(calls, dealer):
    """(trump, declarer, level, doubled) for an auction, or None if it was passed out.

    Mirrors `load_contracts` in bridgezero/play/data.py, which is what the net was
    trained on; keep the two in step.
    """
    bids = [(i, c) for i, c in enumerate(calls) if 0 <= c < PASS]
    if not bids:
        return None
    last_i, last = bids[-1]
    bid_strain = last % 5
    side = (dealer + last_i) % 4 % 2
    first = next(i for i, c in bids if c % 5 == bid_strain and (dealer + i) % 4 % 2 == side)
    tail = calls[last_i + 1:]
    return {
        "trump": BID_STRAIN_TO_TRUMP[bid_strain],
        "declarer": (dealer + first) % 4,
        "level": last // 5 + 1,
        "doubled": 2 if REDOUBLE in tail else (1 if DOUBLE in tail else 0),
    }


def synthetic_auction(level, trump, doubled):
    """The shortest auction that lands in this contract: declarer opens it, all pass."""
    calls = [(level - 1) * 5 + TRUMP_TO_BID_STRAIN[trump]]
    if doubled >= 1:
        calls.append(DOUBLE)
    if doubled == 2:
        calls.append(REDOUBLE)
    return calls + [PASS, PASS, PASS]


def dd_table(owners):
    """(4, 5) double dummy tricks, seat x strain in S H D C NT order. None without a solver."""
    if SOLVER is not None:
        return None
    table = calc_all_tables([Deal.from_pbn(deal_to_pbn(np.asarray(owners)))])[0]
    return np.array([[table[d, p] for d in DENOMS] for p in PLAYERS], dtype=np.int64)


def new_game(owners=None, contract=None, calls=None, dealer=0, vul=(False, False),
             model=None, bench_index=None, bench_dd=None):
    """A board ready to play. `calls` wins over `contract` when both are given."""
    if owners is None:
        owners = deal_owners(np.random.default_rng())
    owners = np.asarray(owners, dtype=np.int64)
    synthetic = False
    if calls:
        found = contract_from_calls(calls, dealer)
        if found is None:
            calls = None
        else:
            contract = found
    if not calls:
        contract = contract or suggested_contract(owners)
        calls = synthetic_auction(contract["level"], contract["trump"], contract["doubled"])
        dealer = contract["declarer"]
        synthetic = True
    return {
        "owners": owners, "calls": list(calls), "dealer": int(dealer), "synthetic": synthetic,
        "trump": int(contract["trump"]), "declarer": int(contract["declarer"]),
        "level": int(contract["level"]), "doubled": int(contract["doubled"]),
        "vul_ns": bool(vul[0]), "vul_ew": bool(vul[1]),
        "played": [], "seat_mode": ["net"] * 4,
        "tricks": dd_table(owners), "bench_index": bench_index, "bench_dd": bench_dd,
        "model": model if model in MODELS else (next(iter(MODELS)) if MODELS else None),
    }


def suggested_contract(owners):
    """A plausible contract for a random deal: the best one double dummy, else 3NT by North."""
    table = dd_table(owners)
    if table is None:
        return {"trump": 4, "declarer": 0, "level": 3, "doubled": 0}
    best, pick = None, (0, 4, 3)
    for seat in range(4):
        for strain in range(5):
            tricks = int(table[seat][strain])
            for level in range(1, 8):
                score = contract_score(level, TRUMP_TO_BID_STRAIN[strain], tricks, 0, False)
                if best is None or score > best:
                    best, pick = score, (seat, strain, level)
    return {"trump": pick[1], "declarer": pick[0], "level": pick[2], "doubled": 0}


def bench():
    """The frozen benchmark, capped, loaded on first use. None when it was not shipped."""
    if BENCH["contracts"] is None and not BENCH["tried"]:
        BENCH["tried"] = True
        if os.path.exists(BENCH_FILE):
            BENCH["contracts"] = load_contracts(BENCH_FILE, BENCH_LIMIT)
    return BENCH["contracts"]


def bench_game(index, model=None):
    rows = bench()
    if rows is None:
        return None
    i = int(index) % len(rows)
    calls = [int(c) for c in rows.calls[i][:int(rows.n_calls[i])]]
    return new_game(
        owners=rows.owner[i].numpy(), calls=calls, dealer=int(rows.dealer[i]),
        vul=(bool(rows.vul_ns[i]), bool(rows.vul_ew[i])), model=model,
        bench_index=i, bench_dd=int(rows.dd_tricks[i]))


def contracts_of(game):
    calls = game["calls"] or [PASS]
    seqs = np.full((1, len(calls)), N_CALLS, dtype=np.int64)
    seqs[0, :len(calls)] = calls
    bag = np.zeros((1, 4, N_CALLS), dtype=np.float32)
    for i, c in enumerate(calls):
        bag[0, (game["dealer"] + i) % 4, c] = 1.0
    return Contracts(
        owner=torch.from_numpy(game["owners"])[None],
        trump=torch.tensor([game["trump"]]),
        declarer=torch.tensor([game["declarer"]]),
        level=torch.tensor([game["level"]]),
        doubled=torch.tensor([game["doubled"]]),
        vul_ns=torch.tensor([game["vul_ns"]]),
        vul_ew=torch.tensor([game["vul_ew"]]),
        call_bag=torch.from_numpy(bag),
        calls=torch.from_numpy(seqs),
        n_calls=torch.tensor([len(calls)]),
        dealer=torch.tensor([game["dealer"]]),
        dd_tricks=torch.tensor([game["bench_dd"] if game["bench_dd"] is not None else -1]),
    )


def batch_of(game, contracts=None):
    c = contracts if contracts is not None else contracts_of(game)
    batch = PlayBatch(c.owner.clone(), c.trump, c.declarer)
    for card in game["played"]:
        batch.play(torch.tensor([card]))
    return batch


def play_card(game, card, batch=None):
    """Append `card` if the seat on turn may play it. False otherwise."""
    if not 0 <= int(card) < 52 or len(game["played"]) >= 52:
        return False
    batch = batch if batch is not None else batch_of(game)
    if not bool(batch.legal()[0, int(card)]):
        return False
    game["played"].append(int(card))
    return True


def net_card(game, contracts=None, batch=None):
    bot = MODELS.get(game["model"])
    if bot is None:
        return None
    c = contracts if contracts is not None else contracts_of(game)
    b = batch if batch is not None else batch_of(game, c)
    step = len(game["played"])
    if game.get("search"):
        # Declarer only; the searcher hands defence back to the net by itself.
        player = bot.searcher()
        player.start(c)
        return int(player.choose(b, c, b.legal(), step)[0])
    return bot.view(c, b, step)["pick"]


# ------------------------------------------------------------- double dummy

def solver_deal(game, upto=None):
    """An endplay Deal wound forward to the position after `upto` cards."""
    deal = Deal(deal_to_pbn(game["owners"]))
    deal.trump = DENOMS[game["trump"]]
    deal.first = PLAYERS[(game["declarer"] + 1) % 4]
    for card in game["played"][:upto]:
        deal.play(Card(suit=DENOMS[card // 13], rank=DDS_RANKS[card % 13]))
    return deal


def solve_here(deal):
    """{card index: tricks the side on turn still takes} at this position."""
    out = {}
    for card, tricks in solve_board(deal):
        out[DENOMS.index(card.suit) * 13 + DDS_RANKS.index(card.rank)] = int(tricks)
    return out


def dd_block(game, batch):
    """What double dummy says about this board, and about the card on turn."""
    declarer_dd = None
    if game["tricks"] is not None:
        declarer_dd = int(game["tricks"][game["declarer"]][game["trump"]])
    block = {
        "available": SOLVER is None,
        "reason": SOLVER and f"no double dummy solver on this server ({SOLVER})",
        "declarer_tricks": declarer_dd,
        "bench_tricks": game["bench_dd"], "bench_index": game["bench_index"],
        "table": game["tricks"].tolist() if game["tricks"] is not None else None,
        "cards": None, "best": None, "best_tricks": None, "side_on_turn": None,
    }
    if SOLVER is not None or batch.done:
        return block
    try:
        per_card = solve_here(solver_deal(game))
    except Exception as exc:                                  # keep the desk alive
        block["available"] = False
        block["reason"] = f"solver failed: {type(exc).__name__}: {exc}"
        return block
    if per_card:
        top = max(per_card.values())
        block["cards"] = {str(k): v for k, v in per_card.items()}
        block["best"] = sorted(k for k, v in per_card.items() if v == top)
        block["best_tricks"] = top
        block["side_on_turn"] = "declaring" if int(batch.to_play()[0]) % 2 == game["declarer"] % 2 \
            else "defending"
    return block


def review(game):
    """Card by card, what double dummy would have taken instead. Needs a finished play."""
    if SOLVER is not None or len(game["played"]) < 52:
        return None
    rows, deal = [], solver_deal(game, upto=0)
    batch = batch_of({**game, "played": []})
    declaring = game["declarer"] % 2
    for step, card in enumerate(game["played"]):
        per_card = solve_here(deal)
        best = max(per_card.values()) if per_card else 0
        got = per_card.get(card, best)
        seat = int(batch.to_play()[0])
        rows.append({
            "step": step, "trick": step // 4, "seat": seat, "card": card,
            "name": card_name(card), "tricks": got, "best": best, "lost": best - got,
            "best_cards": sorted(k for k, v in per_card.items() if v == best),
            "side": "declaring" if seat % 2 == declaring else "defending",
        })
        deal.play(Card(suit=DENOMS[card // 13], rank=DDS_RANKS[card % 13]))
        batch.play(torch.tensor([card]))
    return {
        "cards": rows,
        "declaring_lost": sum(r["lost"] for r in rows if r["side"] == "declaring"),
        "defending_lost": sum(r["lost"] for r in rows if r["side"] == "defending"),
    }


# ------------------------------------------------------------------- state

def state_dump(game):
    contracts = contracts_of(game)
    batch = batch_of(game, contracts)
    step = len(game["played"])
    unplayed = batch.unplayed[0].numpy()
    declarer, dummy = game["declarer"], (game["declarer"] + 2) % 4
    strain = STRAINS[game["trump"]]
    label = f"{game['level']}{strain}" + ["", "X", "XX"][game["doubled"]]

    tricks = []
    for t in range(batch.trick_no):
        leader = int(batch.trick_winner[0, t - 1]) if t else (declarer + 1) % 4
        cards = batch.history[0, t * 4:t * 4 + 4].tolist()
        tricks.append({
            "no": t, "leader": leader, "winner": int(batch.trick_winner[0, t]),
            "cards": [{**card_parts(c), "seat": (leader + k) % 4} for k, c in enumerate(cards)],
        })
    leader = int(batch.trick_winner[0, batch.trick_no - 1]) if batch.trick_no else (declarer + 1) % 4
    current = [{**card_parts(int(c)), "seat": (leader + k) % 4}
               for k, c in enumerate(batch.trick_cards()[0].tolist())]

    ns = int(batch.tricks_won[0, 0])
    ew = int(batch.tricks_won[0, 1])
    made = int(batch.declarer_tricks()[0])
    out = {
        "contract": {
            "level": game["level"], "strain": strain, "trump": game["trump"],
            "doubled": game["doubled"], "label": label,
            "declarer": declarer, "declarer_name": SEAT_NAMES[declarer],
            "dummy": dummy, "dummy_name": SEAT_NAMES[dummy],
            "vul_ns": game["vul_ns"], "vul_ew": game["vul_ew"],
        },
        "auction": {
            "dealer": game["dealer"], "synthetic": game["synthetic"],
            "calls": [{"seat": (game["dealer"] + i) % 4, "call": c, "name": call_name(c)}
                      for i, c in enumerate(game["calls"])],
        },
        "seats": [hand_info(game["owners"], s, unplayed) for s in range(4)],
        "step": step, "done": batch.done,
        "to_play": None if batch.done else int(batch.to_play()[0]),
        "trick_no": batch.trick_no, "trick_leader": leader, "trick": current,
        "tricks": tricks,
        "tricks_won": {"ns": ns, "ew": ew, "declaring": made,
                       "defending": int(batch.tricks_won[0, 1 - declarer % 2])},
        "declarer_tricks": made,
        "seat_mode": game["seat_mode"],
        "model": game["model"], "models": models_list(),
        "bench_size": (len(bench()) if bench() is not None else 0),
        "code": {"d": deal_code(game["owners"]), "p": encode_cards(game["played"]),
                 "a": auction_code(game["calls"]), "dr": game["dealer"],
                 "v": vul_code(game["vul_ns"], game["vul_ew"]),
                 "b": game["bench_index"], "m": game["model"]},
        "dd": dd_block(game, batch),
        "net": None, "legal": None, "review": None,
    }
    if not batch.done:
        out["legal"] = batch.legal()[0].tolist()
        bot = MODELS.get(game["model"])
        if bot is not None:
            out["net"] = bot.view(contracts, batch, step)
    else:
        declarer_vul = game["vul_ns"] if declarer % 2 == 0 else game["vul_ew"]
        score = contract_score(game["level"], TRUMP_TO_BID_STRAIN[game["trump"]],
                               made, game["doubled"], bool(declarer_vul))
        need = game["level"] + 6
        out["result"] = {
            "tricks": made, "needed": need, "delta": made - need,
            "score": score, "made": made >= need,
            "declaring_side": "NS" if declarer % 2 == 0 else "EW",
            "ns_score": score if declarer % 2 == 0 else -score,
        }
        out["review"] = review(game)
    return out


def call_name(c):
    return "Pass" if c == PASS else "X" if c == DOUBLE else "XX" if c == REDOUBLE else NAMES[c]


def deal_code(owners):
    from emergent.bidserver import encode_deal
    return encode_deal(owners)


def auction_code(calls):
    from emergent.bidserver import CALL_CHARS
    return "".join(CALL_CHARS[c] for c in calls)


def vul_code(ns, ew):
    return "all" if ns and ew else "ns" if ns else "ew" if ew else "none"


def vul_of(code):
    return {"none": (False, False), "ns": (True, False),
            "ew": (False, True), "all": (True, True)}.get(str(code), (False, False))


# ------------------------------------------------------------------- routes

def with_game(fn):
    def wrapped():
        gid = request.headers.get("X-Game", "")[:64] or "default"
        with LOCK:
            if gid not in GAMES:
                GAMES[gid] = start_game()
            GAMES.move_to_end(gid)
            while len(GAMES) > MAX_GAMES:
                GAMES.popitem(last=False)
            return fn(GAMES[gid])
    wrapped.__name__ = fn.__name__
    return wrapped


def start_game(model=None):
    """The default board: benchmark row 0 if it shipped, else a random deal."""
    game = bench_game(0, model)
    return game if game is not None else new_game(model=model)


def register(app):
    @app.get("/play")
    def play_page():
        return send_from_directory(STATIC_DIR, "play.html")

    @app.get("/api/play/state")
    @with_game
    def api_play_state(game):
        return jsonify(state_dump(game))

    @app.post("/api/play/setup")
    @with_game
    def api_play_setup(game):
        """Random deal, a benchmark row, or a shared link. Everything is optional."""
        body = request.json or {}
        model = str(body.get("model", "")) or game["model"]
        if body.get("bench") is not None:
            fresh = bench_game(int(body["bench"]), model)
            if fresh is None:
                return jsonify(error="this server has no benchmark file"), 400
        else:
            owners = None
            if body.get("deal"):
                from emergent.bidserver import decode_deal
                owners = decode_deal(str(body["deal"]))
                if owners is None:
                    return jsonify(error="bad deal in link"), 400
            calls = None
            dealer = int(body.get("dealer", 0)) % 4
            if body.get("auction"):
                from emergent.bidserver import CALL_CHARS
                calls = [CALL_CHARS.find(ch) for ch in str(body["auction"])[:250]]
                calls = [c for c in calls if 0 <= c <= REDOUBLE]
            contract = None
            if body.get("contract"):
                c = body["contract"]
                try:
                    contract = {"trump": int(c["trump"]) % 5,
                                "declarer": int(c["declarer"]) % 4,
                                "level": min(7, max(1, int(c["level"]))),
                                "doubled": min(2, max(0, int(c.get("doubled", 0))))}
                except (KeyError, TypeError, ValueError):
                    return jsonify(error="bad contract"), 400
            if calls and contract and body.get("prefer") == "contract":
                calls = None
            fresh = new_game(owners=owners, contract=contract, calls=calls, dealer=dealer,
                             vul=vul_of(body.get("vul", "none")), model=model)
        if body.get("played"):
            for card in decode_cards(body["played"]):
                if not play_card(fresh, card):
                    break
        game.clear()
        game.update(fresh)
        return jsonify(state_dump(game))

    @app.post("/api/play/model")
    @with_game
    def api_play_model(game):
        model = str((request.json or {}).get("model", ""))
        if model not in MODELS:
            return jsonify(error=f"unknown model {model}"), 400
        game["model"] = model
        return jsonify(state_dump(game))

    @app.post("/api/play/seats")
    @with_game
    def api_play_seats(game):
        modes = (request.json or {}).get("seat_mode") or []
        if len(modes) == 4 and all(m in ("net", "you") for m in modes):
            game["seat_mode"] = list(modes)
        return jsonify(state_dump(game))

    @app.post("/api/play/card")
    @with_game
    def api_play_card(game):
        card = int((request.json or {}).get("card", -1))
        if len(game["played"]) >= 52:
            return jsonify(error="the deal is over"), 400
        if not play_card(game, card):
            return jsonify(error=f"{card_name(card) if 0 <= card < 52 else card} "
                                 f"is not a legal card here"), 400
        return jsonify(state_dump(game))

    @app.post("/api/play/net_card")
    @with_game
    def api_play_net_card(game):
        if len(game["played"]) >= 52:
            return jsonify(error="the deal is over"), 400
        card = net_card(game)
        if card is None:
            return jsonify(error="no card-play model is loaded"), 400
        play_card(game, card)
        return jsonify(state_dump(game))

    @app.post("/api/play/auto")
    @with_game
    def api_play_auto(game):
        """Play on while the seat on turn is set to `net`; stop at a seat set to `you`."""
        contracts = contracts_of(game)
        for _ in range(52):
            batch = batch_of(game, contracts)
            if batch.done:
                break
            if game["seat_mode"][int(batch.to_play()[0])] != "net":
                break
            card = net_card(game, contracts, batch)
            if card is None or not play_card(game, card, batch):
                break
        return jsonify(state_dump(game))

    @app.post("/api/play/undo")
    @with_game
    def api_play_undo(game):
        game["played"] = game["played"][:-1]
        return jsonify(state_dump(game))

    @app.post("/api/play/restart")
    @with_game
    def api_play_restart(game):
        game["played"] = []
        return jsonify(state_dump(game))
