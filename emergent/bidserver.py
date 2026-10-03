"""Bid desk server: bid against a checkpoint, real double dummy, the net's full
inputs and outputs for the player on turn.

    python3 -m emergent.bidserver                  # every model in models/models.json
    python3 -m emergent.bidserver --port 8899

Then open http://127.0.0.1:8787. Each browser tab sends its own game id, so
several people can use one server, and each tab picks its own model. Production
runs it under gunicorn, one worker, via `create_app` -- see deploy.sh.

Two model families, each through the exact code it was trained with:

- four-seat bridgezero nets (D, E46; bridgezero/fourseat): policy, Q, trick head,
  double value/gate, 147 or 149 readable input bits;
- brl FSP (emergent/brl_player.py), the external baseline.

The auction rules and all scores come from bridgezero/bridge (AuctionState,
contract_score, dd_par_score). Redouble is never offered: no model has it.
Dealer is North, nobody vulnerable.
"""
import argparse
import functools
import json
import os
import sys
import threading
from collections import OrderedDict

import numpy as np
import torch
from flask import Flask, jsonify, request

from bridgezero.bridge.auction import AuctionState
from bridgezero.bridge.calls import PASS
from bridgezero.bridge.scoring import (contract_score, dd_par_score, own_contract_score,
                                       terminal_ns_score)
from bridgezero.contract.environment import AUCTION_FEATURES
from bridgezero.fourseat.model import (competitive_log_probs, load_fourseat_checkpoint,
                                       policy_log_probs)
from emergent import apis, bench, engine, playdesk, tabledesk
from emergent.deck import (CALL_CHARS, HCP_W, N_CALLS, NAMES, RANKS, SEAT_NAMES, SUITS,
                           TRUMP_TO_BID_STRAIN, call_name, call_token, deal_owners,
                           decode_deal, encode_deal, owners_to_bitmaps, owners_to_pbn)

from endplay.types import Deal, Denom, Player
from endplay.dds import calc_all_tables

L = 35
DENOMS = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs, Denom.nt]  # table order S H D C NT
PLAYERS = [Player.north, Player.east, Player.south, Player.west]
REL_NAMES = ["me", "LHO", "partner", "RHO"]
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")

app = Flask(__name__, static_folder=None)
MODELS = engine.BID_MODELS   # id -> bot
GAMES = OrderedDict()    # game id -> game dict
MAX_GAMES = 300
LOCK = threading.Lock()


# ------------------------------------------------------------------- models
#
# Each family's `decide(hand, calls, dealer, vul, legal)` is its one forward pass and
# its greedy call: engine.choose_call reads it for every page and API, and `view`
# draws the desk's panels from the same pass. `hand` is a (1, 52) float tensor.

def hand_tensor(game, seat):
    return torch.tensor(game["bitmaps"][seat], dtype=torch.float32)[None]


class FourSeatBot:
    """bridgezero four-seat net: greedy over its policy (X via the gate on gated nets)."""

    family = "fourseat"

    def __init__(self, path):
        self.net, ck = load_fourseat_checkpoint(path, "cpu")
        self.net.eval()
        self.step = ck.get("step")
        self.doubles = self.net.n_actions > PASS + 1
        self.competitive = self.net.stage == "D5OWN4XC"
        self.redouble = self.competitive and self.net.redouble
        # X/XX only where Pass would end it, unless the run made them legal everywhere (E46)
        self.final_only = self.competitive and not ck.get("any_seat_double", False)
        self.width = AUCTION_FEATURES + self.net.extra_features
        heads = ["policy", "Q", "tricks"]
        if getattr(self.net, "double_value", False):
            heads.append("double value")
        if getattr(self.net, "gated_double", False):
            heads.append("X gate")
        if self.redouble:
            heads.append("XX gate/value")
        if self.competitive and self.net.sacrifice:
            heads.append("sacrifice gate/value")
        self.info = (f"{ck['stage']} {type(self.net).__name__} · {self.width} input bits · "
                     f"{self.net.n_actions} calls · heads: {', '.join(heads)}")

    # ---- batched helpers, used by emergent/explain.py
    explains = True

    def features(self, history, seats, vul=(False, False)):
        """(B, width) inputs for B rows, each row's own seat on turn (dealer North)."""
        b = len(history)
        return engine.fourseat_features(self, history, torch.zeros(b, dtype=torch.long),
                                        torch.full((b,), float(vul[0])), torch.full((b,), float(vul[1])),
                                        seats)

    def log_probs(self, out, mask):
        return (competitive_log_probs(out, mask) if self.competitive
                else policy_log_probs(out, mask))

    def mask(self, st):
        """This model's legal calls in an AuctionState (X/XX rules can be narrower)."""
        return np.array(engine.legal_calls(self, st))

    @torch.no_grad()
    def decide(self, hand, calls, dealer, vul, legal):
        hist = torch.tensor([calls], dtype=torch.long) if calls else torch.zeros(1, 0, dtype=torch.long)
        seat = (dealer + len(calls)) % 4
        feats = engine.fourseat_features(self, hist, torch.tensor([dealer]),
                                         torch.tensor([float(vul[0])]), torch.tensor([float(vul[1])]),
                                         torch.tensor([seat]))
        out = self.net(hand, feats)
        n = self.net.n_actions
        probs = self.log_probs(out, torch.tensor([legal[:n]]))[0].exp()
        pad = [0.0] * (N_CALLS - n)
        return {"policy": probs.tolist() + pad, "q": out["contract_q"][0].tolist() + pad,
                "pick": int(probs.argmax()), "out": out, "feats": feats}

    @torch.no_grad()
    def batch_log_probs(self, hands, history, dealer, vul_ns, vul_ew, seat, legal):
        """(B, 38) log policy for B positions at once (-inf when illegal): `decide`'s
        numbers, for the bidding search's rollouts. `history` (B, T), -1 padded."""
        feats = engine.fourseat_features(self, history, dealer, vul_ns.float(), vul_ew.float(), seat)
        n = self.net.n_actions
        lp = self.log_probs(self.net(hands, feats), legal[:, :n])
        return torch.cat((lp, torch.full((len(lp), N_CALLS - n), -torch.inf)), 1)

    @torch.no_grad()
    def view(self, game, legal):
        dealer, vul = game.get("dealer", 0), game.get("vul", (False, False))
        seat = (dealer + len(game["calls"])) % 4
        d = self.decide(hand_tensor(game, seat), game["calls"], dealer, vul, legal)
        out, policy, q, pick = d["out"], d["policy"], d["q"], d["pick"]
        f = d["feats"][0].tolist()
        tricks = torch.softmax(out["trick_logits"][0], -1)          # (2 declarers, 5 strains, 14)
        expected = (tricks * torch.arange(14)).sum(-1).tolist()
        heads = {"tricks": {"me": expected[0], "partner": expected[1],
                            "me_seat": seat, "partner_seat": (seat + 2) % 4}}
        if "double_value" in out:
            heads["double_value"] = float(out["double_value"][0])
        if "double_gate" in out:
            heads["p_double"] = float(torch.sigmoid(out["double_gate"][0]))
        if "redouble_gate" in out:
            heads["p_redouble"] = float(torch.sigmoid(out["redouble_gate"][0]))
            heads["redouble_value"] = float(out["redouble_value"][0])
        if "sac_gate" in out:
            cand = out["sac_candidates"][0].tolist()          # strains C D H S NT, -1 = none
            heads["p_sac"] = float(torch.sigmoid(out["sac_gate"][0]))
            heads["sac"] = [{"call": NAMES[c], "value": float(v), "legal": bool(legal[c])}
                            for c, v in zip(cand, out["sac_value"][0].tolist()) if c >= 0]
        segments = [{"name": "my bids", "values": f[0:35]},
                    {"name": "partner bids", "values": f[35:70]},
                    {"name": "passed first", "values": f[70:72]},
                    {"name": "dealer (me/L/P/R)", "values": f[72:76]},
                    {"name": "we are vul", "values": f[76:77]},
                    {"name": "LHO bids", "values": f[77:112]},
                    {"name": "RHO bids", "values": f[112:147]}]
        scalars = [("I passed before we bid", f[70]), ("partner passed before we bid", f[71]),
                   ("dealer is me", f[72]), ("dealer is LHO", f[73]),
                   ("dealer is partner", f[74]), ("dealer is RHO", f[75]), ("we are vulnerable", f[76])]
        if self.doubles:
            segments.append({"name": "doubled flags", "values": f[147:149]})
            scalars += [("contract is doubled", f[147]), ("doubled by my side", f[148])]
        if self.competitive:
            segments.append({"name": "pass ends / XX", "values": f[149:151]})
            scalars += [("my Pass would end it", f[149]), ("contract is redoubled", f[150])]
        return {
            "primary": "policy", "q": q, "policy": policy, "pick": pick,
            "input": {"size": len(f), "scalars": scalars, "segments": segments,
                      "note": "Every input is a plain 0/1 bit. The auction layer is "
                              "base(first 77 bits) + opponent(the rest)."},
            "heads": heads,
        }


class BrlBot:
    """harukaki/brl bidding net (Kita et al., IEEE CoG 2024; Apache 2.0, models/brl_LICENSE):
    PyTorch port verified bit-exact against pgx 1.4.0 + JAX. Greedy over its legal calls, like
    brl's own WBridge5 client. Dealer North, nobody vulnerable, as every model on this desk."""

    family = "brl"
    doubles = redouble = True
    final_only = False                                   # X/XX wherever legal
    SCALARS = ["we not vul", "we vul", "they not vul", "they vul",
               "I passed before any bid", "LHO passed before any bid",
               "partner passed before any bid", "RHO passed before any bid"]

    def __init__(self, path):
        from emergent.brl_player import BrlNet
        self.net = BrlNet(path).eval()
        self.step = "FSP"
        self.info = ("external model: harukaki/brl FSP (Kita et al. 2024, +1.24 IMPs/board vs "
                     "WBridge5) · policy only, no Q or trick head · 480-d pgx input")

    @torch.no_grad()
    def decide(self, hand, calls, dealer, vul, legal):
        from emergent.brl_player import encode, _PGX_TO_OURS
        seat = (dealer + len(calls)) % 4
        hist = torch.tensor([calls], dtype=torch.long) if calls else torch.zeros(1, 0, dtype=torch.long)
        x = encode(hand, hist, torch.tensor([dealer]), torch.tensor([bool(vul[0])]),
                   torch.tensor([bool(vul[1])]), torch.tensor([seat]))
        logits_pgx = self.net(x)[0]
        logits = torch.empty(N_CALLS)
        logits[_PGX_TO_OURS] = logits_pgx                  # our call order: bids, P, X, XX
        mask = torch.tensor(legal[:N_CALLS])
        probs = torch.softmax(logits.masked_fill(~mask, -torch.inf), -1)
        return {"policy": probs.tolist(), "q": logits.tolist(), "pick": int(probs.argmax()), "x": x}

    @torch.no_grad()
    def batch_log_probs(self, hands, history, dealer, vul_ns, vul_ew, seat, legal):
        """(B, 38) log policy for B positions at once (-inf when illegal); see FourSeatBot."""
        from emergent.brl_player import encode, _PGX_TO_OURS
        x = encode(hands, history, dealer, vul_ns.bool(), vul_ew.bool(), seat)
        logits = torch.empty(len(x), N_CALLS)
        logits[:, _PGX_TO_OURS] = self.net(x)
        return torch.log_softmax(logits.masked_fill(~legal[:, :N_CALLS], -torch.inf), -1)

    @torch.no_grad()
    def view(self, game, legal):
        dealer, vul = game.get("dealer", 0), game.get("vul", (False, False))
        seat = (dealer + len(game["calls"])) % 4
        d = self.decide(hand_tensor(game, seat), game["calls"], dealer, vul, legal)
        f = d["x"][0].tolist()
        rel = ["me", "LHO", "partner", "RHO"]
        segments = [{"name": "vulnerability", "values": f[0:4]},
                    {"name": "passed before any bid (me/L/P/R)", "values": f[4:8]}]
        for k, what in enumerate(("bid", "doubled", "redoubled")):
            for r in range(4):
                segments.append({"name": f"{rel[r]} {what}",
                                 "values": [f[8 + 12 * b + 4 * k + r] for b in range(L)]})
        segments.append({"name": "my hand (OpenSpiel card order)", "values": f[428:480]})
        return {
            "primary": "policy", "q": d["q"], "policy": d["policy"], "pick": d["pick"],
            "input": {"size": len(f), "scalars": list(zip(self.SCALARS, f[:8])), "segments": segments,
                      "note": "External brl net: every input is a 0/1 bit of the pgx observation, "
                              "hand included. It has no Q head, so the Q view shows raw policy logits."},
            "heads": None,
        }


def load_models(models_dir=MODELS_DIR):
    with open(os.path.join(models_dir, "models.json")) as fh:
        manifest = json.load(fh)
    for m in manifest:
        path = os.path.join(models_dir, m["file"])
        if m.get("family") == "brl":
            bot = BrlBot(path)
        else:
            bot = FourSeatBot(path)
        bot.id, bot.label, bot.file = m["id"], m["label"], m["file"]
        MODELS[bot.id] = bot


def models_list():
    return [{"id": b.id, "label": b.label, "step": b.step, "doubles": b.doubles,
             "family": b.family, "info": b.info} for b in MODELS.values()]


# ------------------------------------------------------------------- games

def new_game(owners=None, model=None):
    if owners is None:
        owners = deal_owners(np.random.default_rng())
    table = calc_all_tables([Deal.from_pbn(owners_to_pbn(owners))])[0]
    tricks = np.array([[table[d, p] for d in DENOMS] for p in PLAYERS], dtype=np.int64)  # (4,5)
    return {"owners": owners, "bitmaps": owners_to_bitmaps(owners), "tricks": tricks,
            "par": dd_par_score(tricks, False, False), "calls": [],
            "model": model if model in MODELS else engine.default_bid_model()}


def auction(game):
    return AuctionState.from_calls(game["calls"])


def legal_calls(game):
    """38 bools: AuctionState legality, limited to the calls this model can make."""
    return engine.legal_calls(MODELS[game["model"]], auction(game))


def add_call(game, call):
    """Append `call` if legal for this game's model; False otherwise."""
    if auction(game).ended or not 0 <= call < N_CALLS or not legal_calls(game)[call]:
        return False
    game["calls"].append(call)
    return True


def replay(game, calls):
    """Rebuild the auction with as many of `calls` as are legal for the model."""
    game["calls"] = []
    for c in calls:
        if not add_call(game, c):
            break


def with_game(fn):
    @functools.wraps(fn)
    def wrapped():
        gid = request.headers.get("X-Game", "")[:64] or "default"
        with LOCK:
            if gid not in GAMES:
                GAMES[gid] = new_game()
            GAMES.move_to_end(gid)
            while len(GAMES) > MAX_GAMES:
                GAMES.popitem(last=False)
            return fn(GAMES[gid])
    return wrapped


def hand_info(bitmap52):
    sh = bitmap52.reshape(4, 13)
    return {"hcp": int((sh * HCP_W).sum()), "shape": [int(x) for x in sh.sum(1)],
            "cards": {SUITS[s]: [RANKS[r] for r in range(13) if sh[s, r]] for s in range(4)}}


def dd_scores(tricks):
    """Per declarer seat x table strain: tricks and the best undoubled contract with them."""
    cells = []
    for p in range(4):
        row = []
        for d in range(5):
            t, strain = int(tricks[p, d]), TRUMP_TO_BID_STRAIN[d]
            best = None
            for lvl in range(1, 8):
                sc = contract_score(lvl, strain, t, 0, False)
                key = (sc, lvl if sc > 0 else -lvl)
                if best is None or key > best[0]:
                    best = (key, (lvl - 1) * 5 + strain, sc)
            row.append({"tricks": t, "contract": NAMES[best[1]], "score": best[2]})
        cells.append(row)
    sides = {}
    for side, seats in (("NS", (0, 2)), ("EW", (1, 3))):
        b = max((cells[p][d]["score"], p, d) for p in seats for d in range(5))
        sides[side] = {"score": b[0], "seat": SEAT_NAMES[b[1]], "contract": cells[b[1]][b[2]]["contract"]}
    return {"cells": cells, "best": sides}


def state_dump(game):
    st = auction(game)
    over = st.ended
    bot = MODELS[game["model"]]
    out = {
        "turn": len(game["calls"]), "current_seat": None if over else st.turn, "game_over": over,
        "seats": [{"seat": s, "name": SEAT_NAMES[s], **hand_info(game["bitmaps"][s])} for s in range(4)],
        "auction": [{"seat": i % 4, "call": c, "name": call_name(c)} for i, c in enumerate(game["calls"])],
        "net": None, "dd": dd_scores(game["tricks"]), "names": NAMES,
        "deal_code": encode_deal(game["owners"]),
        "auction_code": "".join(CALL_CHARS[c] for c in game["calls"]),
        "par": game["par"], "tricks_all": game["tricks"].tolist(),
        "model": bot.id, "models": models_list(),
    }
    if not over:
        legal = legal_calls(game)
        view = bot.view(game, legal)
        out["net"] = {"seat": st.turn, "legal": legal, "pick_name": call_name(view["pick"]), **view}
        out["why"] = why(view, legal)
        out["meaning"] = meaning(game)
        out["explains"] = bool(getattr(bot, "explains", False))
    else:
        tricks = game["tricks"]
        out["contract"] = None if st.last_contract < 0 else NAMES[st.last_contract]
        out["doubled"] = st.doubled                                    # 0, 1 = X, 2 = XX
        out["declarer"] = None if st.last_contract < 0 else SEAT_NAMES[st.declarer()]
        out["ns_score"] = terminal_ns_score(st, tricks)
        out["gap_to_par"] = out["ns_score"] - game["par"]
        out["ns_own_score"] = own_contract_score(st, tricks, 0)
        out["ew_own_score"] = own_contract_score(st, tricks, 1)
    return out


def net_pick(game):
    seat = len(game["calls"]) % 4
    return engine.choose_call(MODELS[game["model"]], game["bitmaps"][seat], game["calls"])[0]


# ------------------------------------------------------------------- routes

@app.get("/api/state")
@with_game
def api_state(game):
    return jsonify(state_dump(game))


@app.post("/api/new_deal")
@with_game
def api_new_deal(game):
    game.update(new_game(model=game["model"]))
    return jsonify(state_dump(game))


@app.post("/api/model")
@with_game
def api_model(game):
    model = str(request.json.get("model", ""))
    if model not in MODELS:
        return jsonify(error=f"unknown model {model}"), 400
    game["model"] = model
    replay(game, list(game["calls"]))   # drops a Double the new model cannot make, and what follows
    return jsonify(state_dump(game))


@app.post("/api/call")
@with_game
def api_call(game):
    call = int(request.json["call"])
    if auction(game).ended:
        return jsonify(error="auction is over"), 400
    if not add_call(game, call):
        return jsonify(error=f"{call_name(call)} is not legal for {SEAT_NAMES[auction(game).turn]}"), 400
    return jsonify(state_dump(game))


@app.post("/api/net_call")
@with_game
def api_net_call(game):
    if auction(game).ended:
        return jsonify(error="auction is over"), 400
    add_call(game, net_pick(game))
    return jsonify(state_dump(game))


@app.post("/api/auto_end")
@with_game
def api_auto_end(game):
    guard = 0
    while not auction(game).ended and guard < 250:
        add_call(game, net_pick(game))
        guard += 1
    return jsonify(state_dump(game))


@app.post("/api/load")
@with_game
def api_load(game):
    """Open a shared link: model, deal, then as many of its calls as are legal."""
    owners = decode_deal(str(request.json.get("deal", "")))
    if owners is None:
        return jsonify(error="bad deal in link"), 400
    model = str(request.json.get("model", "")) or game["model"]
    game.update(new_game(owners, model if model in MODELS else game["model"]))
    replay(game, [CALL_CHARS.find(ch) for ch in str(request.json.get("auction", ""))[:250]])
    return jsonify(state_dump(game))


@app.post("/api/undo")
@with_game
def api_undo(game):
    replay(game, game["calls"][:-1])
    return jsonify(state_dump(game))


@app.post("/api/restart")
@with_game
def api_restart(game):
    replay(game, [])
    return jsonify(state_dump(game))


# ------------------------------------------------------- why did it bid that

CORPUS = {}                  # model id -> {prefix: {"n": int, "calls": {name: stats}}}
EXPLAIN = {"lock": threading.Lock(), "jobs": OrderedDict(), "next": None, "worker": None}
MAX_EXPLAIN_JOBS = 40


def load_corpus(models_dir=MODELS_DIR):
    """models/corpus_<model id>.json: what each call meant in offline self-play."""
    for bot_id in MODELS:
        path = os.path.join(models_dir, f"corpus_{bot_id}.json")
        if os.path.exists(path):
            with open(path) as fh:
                CORPUS[bot_id] = json.load(fh)


def meaning(game):
    """The offline table for this exact position, if self-play ever reached it."""
    table = CORPUS.get(game["model"])
    if table is None:
        return None
    key = "-".join(call_token(c) for c in game["calls"])
    hit = table.get(key)
    return None if hit is None else {"position": key or "(opening)", **hit}


def why(view, legal):
    """The net's own numbers for its call and the runner-up: chance and score guess."""
    policy, q = view.get("policy"), view.get("q")
    if not policy:
        return None
    ranked = sorted((p, c) for c, p in enumerate(policy) if legal[c])[::-1][:2]
    rows = [{"call": call_name(c), "p": round(p, 3), "q": round(q[c] * 100, 0) if q else None}
            for p, c in ranked]
    return {"calls": rows, "note": "q is how far below the best possible contract the net expects "
                                   "to finish after that call, in points (0 = perfect, less is worse)."}


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
            bot = MODELS[snapshot["model"]]
            result = run_explain(bot, snapshot, snapshot["candidates"], snapshot["samples"])
            entry = {"status": "done", "result": result}
        except Exception as exc:                          # keep the desk alive on any failure
            entry = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        with EXPLAIN["lock"]:
            EXPLAIN["jobs"][key] = entry
            while len(EXPLAIN["jobs"]) > MAX_EXPLAIN_JOBS:
                EXPLAIN["jobs"].popitem(last=False)


@app.post("/api/explain")
@with_game
def api_explain(game):
    """Queue a rollout for the position on turn; the newest request wins. Returns a job key."""
    bot = MODELS[game["model"]]
    if not getattr(bot, "explains", False):
        return jsonify(error="this model cannot be rolled out"), 400
    st = auction(game)
    if st.ended:
        return jsonify(error="auction is over"), 400
    samples = max(32, min(512, int((request.json or {}).get("samples", 256))))
    legal = legal_calls(game)
    view = bot.view(game, legal)
    ranked = sorted((p, c) for c, p in enumerate(view["policy"] or []) if legal[c])[::-1][:2]
    candidates = [c for _, c in ranked]
    key = f"{game['model']}|{encode_deal(game['owners'])}|" \
          f"{''.join(CALL_CHARS[c] for c in game['calls'])}|{samples}"
    snapshot = {"model": game["model"], "calls": list(game["calls"]),
                "bitmaps": game["bitmaps"], "candidates": candidates, "samples": samples}
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


@app.get("/api/explain")
def api_explain_result():
    key = request.args.get("job", "")
    with EXPLAIN["lock"]:
        entry = EXPLAIN["jobs"].get(key)
    return jsonify(job=key, **(entry or {"status": "unknown"}))


playdesk.register(app)          # /api/play/*, the card play desk's API (its page is gone)
tabledesk.register(app)         # /table and /api/table/*, play a board against the nets
apis.register(app)              # /apis/bbo.php and /apis/brill/*, for other sites' tables
bench.register(app)             # /bench and /apis/bench/*, weak-spot reports for anyone's bot


def create_app(ckpt=None):
    """gunicorn entry point. `ckpt` is ignored (old unit files pass 'ck.pt'); models/models.json rules."""
    torch.set_num_threads(1)
    load_models()
    load_corpus()
    playdesk.load_models()
    tabledesk.load(sys.modules[__name__])
    apis.load(sys.modules[__name__])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    torch.set_num_threads(1)
    load_models()
    load_corpus()
    playdesk.load_models()
    tabledesk.load(sys.modules[__name__])
    apis.load(sys.modules[__name__])
    for b in MODELS.values():
        print(f"  {b.id:10s} step {b.step}  {b.info}")
    for b in playdesk.MODELS.values():
        print(f"  {b.id:10s} play    {b.info}")
    print(f"open http://{a.host}:{a.port}   (the table; /table works too)")
    app.run(host=a.host, port=a.port, debug=False)


if __name__ == "__main__":
    main()
