"""Turn one bidding decision into beginner facts, and match it against D's teaching rules.

A decision is (calls so far, dealer, seat to act, 13-card hand). This module abstracts it into

* a SITUATION (who opened, what partner and I have said, is there a fit, who competes, level),
* HAND FACTS (HCP, support for partner, longest suit, balanced, cards in the opponents' suit),
* an ACTION TYPE for any call, relative to the situation (pass, support, new suit, rebid own
  suit, notrump, double, bid their suit; each bid also gets a level kind: min / jump / game / slam),

and applies the rules in rules.json to suggest what D would do. numpy is the only dependency.

Conventions (same as training): calls 0..34 are 1C,1D,1H,1S,1NT,2C,...,7NT; 35 Pass, 36 X,
37 XX. Seats 0..3 = N E S W. A card is suit * 13 + rank with suits S H D C and rank 0 = ace.

    from situations import advise, load_rules, parse_hand, parse_calls
    rules = load_rules()                                  # rules.json next to this file
    out = advise(parse_calls("1H P"), dealer=0, seat=2, hand=parse_hand("KQ52.J84.A73.962"),
                 rules=rules)
    out["call_name"], out["rule"]["title"], out["branch"]["say"]

Self-test (no data needed):  python situations.py
"""
from __future__ import annotations

import json
import os

import numpy as np

STRAINS = ("C", "D", "H", "S", "NT")
CALL_NAMES = tuple(f"{lv}{s}" for lv in range(1, 8) for s in STRAINS) + ("P", "X", "XX")
PASS, DOUBLE, REDOUBLE = 35, 36, 37
HAND_SUITS = "SHDC"                        # card // 13 order
RANKS = "AKQJT98765432"
STRAIN_TO_SUIT = {0: 3, 1: 2, 2: 1, 3: 0}  # bid strain C,D,H,S -> hand suit index
SUIT_TO_STRAIN = {v: k for k, v in STRAIN_TO_SUIT.items()}
REL = ("me", "lho", "partner", "rho")      # relative seat = (absolute - my seat) % 4
GAME_LEVEL = {0: 5, 1: 5, 2: 4, 3: 4, 4: 3}  # by strain C D H S NT
LEVEL_KINDS = ("min", "jump", "game", "slam")
ACTION_TYPES = ("pass", "double", "redouble", "support", "rebid", "new_suit", "nt", "their_suit")
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- parsing helpers
def parse_calls(text):
    """'1H P 1NT X' -> [2, 35, 4, 36]."""
    out = []
    for tok in text.replace("-", " ").split():
        t = tok.upper()
        t = {"PASS": "P", "DBL": "X", "RDBL": "XX"}.get(t, t)
        if len(t) == 2 and t[1] == "N":
            t = t[0] + "NT"
        out.append(CALL_NAMES.index(t))
    return out


def parse_hand(text):
    """'KQ52.J84.A73.962' (S.H.D.C) -> list of 13 card ids."""
    parts = text.upper().replace("10", "T").split(".")
    assert len(parts) == 4, text
    cards = [s * 13 + RANKS.index(ch) for s, p in enumerate(parts) for ch in p if ch != "-"]
    assert len(cards) == 13, text
    return cards


def call_name(c):
    return CALL_NAMES[c]


# ----------------------------------------------------------------------------- hand facts
def hand_facts(cards):
    cards = np.asarray(sorted(int(c) for c in cards))
    suit, rank = cards // 13, cards % 13
    lens = np.bincount(suit, minlength=4)
    hcp_s = np.bincount(suit, weights=np.maximum(0, 4 - rank), minlength=4).astype(int)
    srt = sorted(lens, reverse=True)
    bal = (srt[0] <= 4 or (srt[0] == 5 and srt[1] == 3)) and srt[3] >= 2
    return {"hcp": int(hcp_s.sum()), "len": [int(x) for x in lens], "suit_hcp": hcp_s.tolist(),
            "balanced": bool(bal), "longest": int(srt[0]), "shortest": int(srt[3]),
            "shape": "".join(map(str, srt))}


# ----------------------------------------------------------------------------- auction facts
def auction_facts(calls, dealer, seat):
    """Relative view of the auction so far for `seat` (who is about to call)."""
    info = {r: {"calls": [], "bids": [], "suits": {}, "suit_order": [], "nt": 0, "x": 0} for r in REL}
    opener = None
    open_idx = None
    contract = None            # [level, strain, rel owner, doubled 0/1/2]
    for i, c in enumerate(calls):
        rel = REL[(dealer + i - seat) % 4]
        p = info[rel]
        p["calls"].append(c)
        if c < 35:
            lv, st = c // 5 + 1, c % 5
            p["bids"].append(c)
            if st == 4:
                p["nt"] += 1
            else:
                s = STRAIN_TO_SUIT[st]
                p["suits"][s] = p["suits"].get(s, 0) + 1
                p["suit_order"].append(s)
            if opener is None:
                opener, open_idx = rel, i
            contract = [lv, st, rel, 0]
        elif c == DOUBLE:
            p["x"] += 1
            contract[3] = 1
        elif c == REDOUBLE:
            contract[3] = 2
    turns = {r: 0 for r in REL}
    if open_idx is not None:
        for i in range(open_idx + 1, len(calls)):
            turns[REL[(dealer + i - seat) % 4]] += 1
    return {"info": info, "opener": opener, "contract": contract, "n": len(calls),
            "turns_after_open": turns}


def legal_min_level(contract, strain):
    if contract is None:
        return 1
    lv, st = contract[0], contract[1]
    return lv if strain > st else lv + 1


def level_kind(level, strain, contract):
    if level >= 6:
        return "slam"
    if level >= GAME_LEVEL[strain]:
        return "game"
    return "min" if level == legal_min_level(contract, strain) else "jump"


def action_type(call, af):
    """(type, level kind) of `call` relative to auction facts `af` (taken before the call)."""
    if call == PASS:
        return "pass", None
    if call == DOUBLE:
        return "double", None
    if call == REDOUBLE:
        return "redouble", None
    info = af["info"]
    lv, st = call // 5 + 1, call % 5
    lk = level_kind(lv, st, af["contract"])
    if st == 4:
        return "nt", lk
    s = STRAIN_TO_SUIT[st]
    if s in info["partner"]["suits"]:
        return "support", lk
    if s in info["me"]["suits"]:
        return "rebid", lk
    if s in info["lho"]["suits"] or s in info["rho"]["suits"]:
        return "their_suit", lk
    return "new_suit", lk


def act_label(call, af):
    at, lk = action_type(call, af)
    return at + ("_" + lk if lk else "")


def opening_label(call):
    if call == PASS:
        return "pass"
    if call % 5 == 4:
        return "open_nt"
    return "open_suit1" if call < 5 else "open_high"


# ----------------------------------------------------------------------------- situations
SITUATIONS = {
    "R_1suit": "Partner opened one of a suit; my first turn",
    "R_1NT": "Partner opened 1NT (strong, any shape); my first turn",
    "O_rebid": "I opened, partner has bid; my second turn",
    "O_alone": "I opened, partner only passed, opponents compete; my second turn",
    "R_rebid": "Partner opened, I have had one turn; my second turn",
    "WE_later": "We opened; my third turn or later",
    "D_direct": "Right-hand opponent opened; my first turn",
    "D_lho": "Left-hand opponent opened, partner passed; my first turn",
    "D_advance": "They opened, partner has bid or doubled, I have not bid yet",
    "D_silent": "They opened, neither partner nor I have bid, my second turn or later",
    "D_later": "They opened, I have already bid or doubled",
}


FEATURES = {
    # situation facts (from the auction only)
    "sit": "coarse situation id, see SITUATIONS",
    "we_opened": "our side made the first bid",
    "rho": "right-hand opponent's last call: pass / bid / double / redouble / none",
    "contested": "the other side has made a non-pass call",
    "opp_bids": "number of bids (not passes or doubles) by the opponents",
    "level": "level of the current contract (0 = none)",
    "owner": "who holds the current contract: us / them",
    "doubled": "current contract is doubled (1) or redoubled (2)",
    "game_reached": "current contract is at game level or higher (3NT, 4H/4S, 5C/5D)",
    "fit": "partner and I have both bid the same suit",
    "partner_suit": "partner has bid a suit",
    "partner_nt": "partner has bid notrump",
    "partner_doubled": "partner has doubled",
    "partner_bids": "number of partner's non-pass calls",
    "my_bids": "number of my non-pass calls",
    "my_suit": "I have bid a suit",
    "partner_last": "action label of partner's last informative call (e.g. support_min, nt_min, pass)",
    "partner_key": "meaning key 'situation|action' of partner's last informative call",
    "partner_last_type": "type of partner's last informative call: support / new_suit / rebid / nt / pass / double / their_suit",
    "partner_last_level": "level kind of partner's last call: min / jump / game / slam / ''",
    "partner_raised_me": "partner's last suit bid was in a suit I bid first or also bid",
    "partner_lo": "HCP partner has shown: 10th percentile of D's hands for that call",
    "partner_mid": "HCP partner has shown: median",
    "partner_hi": "HCP partner has shown: 90th percentile",
    "my_shown": "median HCP D holds for my own last informative call (what partner thinks I have)",
    # hand facts
    "hcp": "high card points (A4 K3 Q2 J1)",
    "balanced": "4333, 4432 or 5332",
    "longest": "length of my longest suit",
    "shortest": "length of my shortest suit",
    "support": "my length in partner's suit (the partner suit I hold most of)",
    "own_len": "length of my longest suit that I have already bid",
    "new_len": "length of the new suit I would bid (see _pick_new_suit)",
    "new_level": "level at which that new suit would be bid (0 = none possible)",
    "suit_margin": "suit test: HCP + 3 x new_len - (18 + 3 x new_level); >= 0 passes",
    "their_len": "my length in the opponents' last bid suit (-1 = they bid no suit)",
    "their_hcp": "my HCP in the opponents' last bid suit",
    "their_stopper": "A, Kx, Qxx or Jxxx in the opponents' last bid suit",
    "fit_cards": "most cards partner and I hold together in one of partner's suits (partner counted 4 per suit, +1 per rebid, 3 for raising mine)",
    "total": "hcp + partner_mid",
    "extra": "hcp - my_shown (points beyond what I have already shown)",
}


def situation(af):
    """Coarse situation id for the player about to call (see SITUATIONS)."""
    op = af["opener"]
    info = af["info"]
    if op is None:
        return "pre_open"
    me, pa = info["me"], info["partner"]
    my_turns = af["turns_after_open"]["me"]
    my_n = sum(c != PASS for c in me["calls"])
    pa_n = sum(c != PASS for c in pa["calls"])
    if op in ("me", "partner"):
        if op == "partner" and my_turns == 0:
            return "R_1NT" if pa["bids"][0] % 5 == 4 else "R_1suit"
        if op == "me" and my_turns == 0:
            return "O_rebid" if pa_n else "O_alone"
        if op == "partner" and my_turns == 1:
            return "R_rebid"
        return "WE_later"
    if my_turns == 0 and pa_n == 0:
        return "D_direct" if op == "rho" else "D_lho"
    if my_n == 0 and pa_n > 0:
        return "D_advance"
    if my_n == 0:
        return "D_silent"
    return "D_later"


def call_keys(calls, dealer=0):
    """Meaning key 'situation|action' of every call in an auction (in order)."""
    keys = []
    for t, c in enumerate(calls):
        af = auction_facts(calls[:t], dealer, (dealer + t) % 4)
        s = situation(af)
        keys.append("open|" + opening_label(c) if s == "pre_open" else s + "|" + act_label(c, af))
    return keys


# ----------------------------------------------------------------------------- decision facts
def _pick_suit(cands, lens, prefer):
    """Longest suit among cands; ties broken by `prefer` order (first listed wins)."""
    if not cands:
        return None
    best = max(lens[s] for s in cands)
    tied = [s for s in cands if lens[s] == best]
    for s in prefer:
        if s in tied:
            return s
    return tied[0]


def _pick_new_suit(unbid, lens, contract):
    """D's choice of new suit (93% agreement): a 4+ card suit you can still show at the
    1-level comes first; otherwise the longest suit. Ties: two 4-card suits -> the cheaper
    bid, two 5+ card suits -> the higher-ranking suit."""
    if not unbid:
        return None
    lv = {s: legal_min_level(contract, SUIT_TO_STRAIN[s]) for s in unbid}

    def best(group):
        mx = max(lens[s] for s in group)
        tied = [s for s in group if lens[s] == mx]
        if mx <= 4:
            return min(tied, key=lambda s: (lv[s], SUIT_TO_STRAIN[s]))
        return min(tied)                      # suit index 0 = spades = highest ranking
    one = [s for s in unbid if lens[s] >= 4 and lv[s] == 1]
    return best(one) if one else best(unbid)


def _bid_at(suit_or_nt, kind, contract):
    """Concrete call for a strain (hand suit index or 'NT') at a level kind, or None."""
    st = 4 if suit_or_nt == "NT" else SUIT_TO_STRAIN[suit_or_nt]
    lo = legal_min_level(contract, st)
    lv = {"min": lo, "jump": lo + 1, "game": max(lo, GAME_LEVEL[st]), "slam": max(lo, 6)}[kind]
    if lv > 7:
        return None
    return (lv - 1) * 5 + st


def _act_type(act):
    """'support_min' -> 'support'; 'open_nt' -> 'nt'; 'open_suit1' -> 'new_suit'; 'pass' -> 'pass'."""
    if act == "open_nt":
        return "nt"
    if act.startswith("open"):
        return "new_suit"
    for t in ACTION_TYPES:
        if act.startswith(t):
            return t
    return act


def _act_level(act):
    """Level kind of an action label ('min', 'jump', 'game', 'slam') or '' for pass / double."""
    for k in LEVEL_KINDS:
        if act.endswith("_" + k):
            return k
    return ""


def _first(calls, dealer, seat, suit):
    """Relative seat of whoever bid `suit` first (None if nobody)."""
    for i, c in enumerate(calls):
        if c < 35 and c % 5 != 4 and STRAIN_TO_SUIT[c % 5] == suit:
            return REL[(dealer + i - seat) % 4]
    return None


def _stopper(cards, suit):
    """A, Kx, Qxx or Jxxx in `suit`."""
    ranks = sorted(int(c) % 13 for c in cards if int(c) // 13 == suit)
    n = len(ranks)
    return any((r == 0) or (r == 1 and n >= 2) or (r == 2 and n >= 3) or (r == 3 and n >= 4) for r in ranks)


def decision_facts(calls, dealer, seat, cards, meanings=None, keys=None):
    """All features the rules may test, plus the candidate call for every action spec.

    meanings: {'situation|action': {'lo','mid','hi'}} = HCP of the caller (rules.json
    'call_meanings'); gives partner_lo/mid/hi, my_shown, total and extra.
    keys: optional precomputed call_keys(calls, dealer) (speed only).
    """
    calls = [int(c) for c in calls]
    af = auction_facts(calls, dealer, seat)
    info = af["info"]
    me, pa, lho, rho = info["me"], info["partner"], info["lho"], info["rho"]
    h = hand_facts(cards)
    L = h["len"]
    sit = situation(af)
    k = af["contract"]
    bid_suits = set(me["suits"]) | set(pa["suits"]) | set(lho["suits"]) | set(rho["suits"])
    unbid = [s for s in range(4) if s not in bid_suits]
    opp_order = [STRAIN_TO_SUIT[c % 5] for i, c in enumerate(calls)
                 if c < 35 and c % 5 != 4 and REL[(dealer + i - seat) % 4] in ("lho", "rho")]
    their = opp_order[-1] if opp_order else None        # suit of their last suit bid
    p_recent = list(dict.fromkeys(reversed(pa["suit_order"])))
    sup_suit = _pick_suit(p_recent, L, p_recent)        # partner's suit I hold most of
    own_suit = _pick_suit(list(me["suits"]), L, list(dict.fromkeys(reversed(me["suit_order"]))))
    new_suit = _pick_new_suit(unbid, L, k)
    if meanings is not None and keys is None:
        keys = call_keys(calls, dealer)

    def last_key(rel):
        idx = [i for i in range(len(calls)) if REL[(dealer + i - seat) % 4] == rel]
        if not idx or keys is None:
            return "none"
        np_ = [i for i in idx if calls[i] != PASS]
        return keys[np_[-1] if np_ else idx[-1]]
    pkey, mkey = last_key("partner"), last_key("me")
    M = meanings or {}
    dflt = {"lo": 5, "mid": 10, "hi": 15}
    pm, mm = M.get(pkey, dflt), M.get(mkey, dflt)
    rho_last = rho["calls"][-1] if rho["calls"] else None
    we = af["opener"] in ("me", "partner")
    f = {
        # ---- situation facts
        "sit": sit,
        "we_opened": we,
        "rho": "none" if rho_last is None else {PASS: "pass", DOUBLE: "double",
                                                 REDOUBLE: "redouble"}.get(rho_last, "bid"),
        "contested": any(c != PASS for c in lho["calls"] + rho["calls"]) if we
                     else any(c != PASS for c in me["calls"] + pa["calls"]),
        "opp_bids": len(lho["bids"]) + len(rho["bids"]),
        "level": 0 if k is None else k[0],
        "owner": "none" if k is None else ("us" if k[2] in ("me", "partner") else "them"),
        "doubled": 0 if k is None else k[3],
        "game_reached": bool(k is not None and k[0] >= GAME_LEVEL[k[1]]),
        "fit": any(s in me["suits"] for s in pa["suits"]),
        "partner_suit": bool(pa["suits"]),
        "partner_nt": pa["nt"] > 0,
        "partner_doubled": pa["x"] > 0,
        "partner_bids": sum(c != PASS for c in pa["calls"]),
        "my_bids": sum(c != PASS for c in me["calls"]),
        "my_suit": bool(me["suits"]),
        "partner_last": pkey.split("|")[-1],
        "partner_key": pkey,
        "partner_last_type": _act_type(pkey.split("|")[-1]),
        "partner_last_level": _act_level(pkey.split("|")[-1]),
        "partner_raised_me": bool(pa["suit_order"]) and pa["suit_order"][-1] in me["suits"],
        "partner_lo": pm["lo"], "partner_mid": pm["mid"], "partner_hi": pm["hi"],
        "my_shown": mm["mid"],
        # ---- hand facts
        "hcp": h["hcp"],
        "balanced": h["balanced"],
        "longest": h["longest"],
        "shortest": h["shortest"],
        "support": L[sup_suit] if sup_suit is not None else 0,
        "own_len": L[own_suit] if own_suit is not None else 0,
        "new_len": L[new_suit] if new_suit is not None else 0,
        "their_len": L[their] if their is not None else -1,
        "their_hcp": h["suit_hcp"][their] if their is not None else -1,
    }
    # cards partner has promised per suit: 4 for a suit partner introduced, 3 for raising my
    # suit, +1 for every repeat; fit_cards = my length + that, best suit
    promised = {s: (3 if _first(calls, dealer, seat, s) == "me" else 4) + n - 1
                for s, n in pa["suits"].items()}
    f["fit_cards"] = max([L[s] + n for s, n in promised.items()], default=0)
    ts = their
    f["their_stopper"] = bool(ts is not None and _stopper(cards, ts))
    nc = _bid_at(new_suit, "min", k) if new_suit is not None else None
    f["new_level"] = 0 if nc is None else nc // 5 + 1   # level my new suit would be bid at
    # "suit points" test for showing a new suit in competition: HCP + 3 x length must reach
    # 18 + 3 x the level you bid at (21 at the 1-level, 24 at the 2-level, 27 at the 3-level)
    f["suit_margin"] = (f["hcp"] + 3 * f["new_len"] - 18 - 3 * f["new_level"]) if nc is not None else -99
    f["total"] = f["hcp"] + f["partner_mid"]
    f["extra"] = f["hcp"] - f["my_shown"]
    cand = {"pass": PASS}
    cand["double"] = DOUBLE if (k is not None and k[2] in ("lho", "rho") and k[3] == 0) else None
    cand["redouble"] = REDOUBLE if (k is not None and k[2] in ("me", "partner") and k[3] == 1) else None
    for name, s in (("support", sup_suit), ("rebid", own_suit), ("new_suit", new_suit),
                    ("their_suit", their), ("nt", "NT")):
        for kind in LEVEL_KINDS:
            cand[f"{name}_{kind}"] = None if s is None else _bid_at(s, kind, k)
    f["_cand"] = cand
    f["_af"] = af
    return f


# ----------------------------------------------------------------------------- rule matching
def cond_ok(spec, value):
    """One condition. spec: [lo, hi] numeric inclusive (None = open end), bool, str, or list of str."""
    if isinstance(spec, list) and len(spec) == 2 and not any(isinstance(x, (bool, str)) for x in spec):
        lo, hi = spec
        return (lo is None or value >= lo) and (hi is None or value <= hi)
    if isinstance(spec, list):
        return value in spec
    return value == spec


def conds_ok(conds, f):
    """All conditions hold. Special keys: 'any': [conds, ...] (one must hold), 'not': conds."""
    for key, spec in conds.items():
        if key == "any":
            if not any(conds_ok(c, f) for c in spec):
                return False
        elif key == "not":
            if conds_ok(spec, f):
                return False
        elif not cond_ok(spec, f[key]):
            return False
    return True


def resolve(do, f):
    """Action spec {'type': ..., 'level': 'min'|'jump'|'game'|'slam'|int} -> call id, or None."""
    t = do["type"]
    if t in ("pass", "double", "redouble"):
        return f["_cand"][t]
    lv = do.get("level", "min")
    if isinstance(lv, int):                    # an exact level, e.g. 4NT
        c = f["_cand"].get(f"{t}_min")
        if c is None or lv < c // 5 + 1 or lv > 7:
            return None
        return (lv - 1) * 5 + c % 5
    return f["_cand"].get(f"{t}_{lv}")


def advise(calls, dealer, seat, hand, rules, facts=None):
    """First rule whose situation matches; inside it the first branch whose hand conditions
    match and whose action is possible. Returns {'call', 'call_name', 'rule', 'branch',
    'facts'} or None when no rule covers the decision. `hand` = 13 card ids."""
    f = facts or decision_facts(calls, dealer, seat, hand, rules.get("call_meanings"))
    for rule in rules["rules"]:
        if not conds_ok(rule["situation"], f):
            continue
        for br in rule["branches"]:
            if conds_ok(br.get("when", {}), f):
                c = resolve(br["do"], f)
                if c is None:
                    continue
                return {"call": c, "call_name": CALL_NAMES[c], "rule": rule, "branch": br, "facts": f}
        return None
    return None


def load_rules(path=None):
    with open(path or os.path.join(HERE, "rules.json")) as fh:
        return json.load(fh)


def partner_shown(calls, dealer, seat, rules):
    """What partner's last informative call showed: {'call', 'key', 'lo', 'mid', 'hi', 'text'?}."""
    keys = call_keys(calls, dealer)
    idx = [i for i in range(len(calls)) if (dealer + i - seat) % 4 == 2]
    if not idx:
        return None
    np_ = [i for i in idx if calls[i] != PASS]
    i = np_[-1] if np_ else idx[-1]
    m = rules["call_meanings"].get(keys[i])
    return None if m is None else {"call": CALL_NAMES[calls[i]], "key": keys[i], **m}


# ----------------------------------------------------------------------------- self-test
def _selftest():
    assert parse_calls("1H P 1NT X") == [2, 35, 4, 36]
    assert parse_calls("pass 2nt xx") == [35, 9, 37]
    h = parse_hand("KQ52.J84.A73.962")
    assert hand_facts(h)["hcp"] == 10 and hand_facts(h)["balanced"]
    af = auction_facts(parse_calls("1H P"), 0, 2)             # N opens 1H, E passes, S to call
    assert situation(af) == "R_1suit"
    assert action_type(parse_calls("2H")[0], af) == ("support", "min")
    assert action_type(parse_calls("3H")[0], af) == ("support", "jump")
    assert action_type(parse_calls("4H")[0], af) == ("support", "game")
    assert action_type(parse_calls("1S")[0], af) == ("new_suit", "min")
    assert action_type(parse_calls("1NT")[0], af) == ("nt", "min")
    assert situation(auction_facts(parse_calls("P 1C"), 0, 2)) == "D_direct"
    assert situation(auction_facts(parse_calls("1C P 1H"), 0, 3)) == "D_lho"
    assert situation(auction_facts(parse_calls("1C 1S P"), 0, 3)) == "D_advance"
    assert situation(auction_facts(parse_calls("1C P 1H P"), 0, 0)) == "O_rebid"
    assert situation(auction_facts(parse_calls("1C P P 1S"), 0, 0)) == "O_alone"
    assert situation(auction_facts(parse_calls("1C P 1H P 1NT P"), 0, 2)) == "R_rebid"
    f = decision_facts(parse_calls("1H P"), 0, 2, h)
    assert f["support"] == 3 and f["new_len"] == 4
    assert CALL_NAMES[f["_cand"]["support_min"]] == "2H" and CALL_NAMES[f["_cand"]["new_suit_min"]] == "1S"
    f = decision_facts(parse_calls("P 1C"), 0, 2, h)
    assert f["their_len"] == 3 and f["_cand"]["double"] == DOUBLE and f["_cand"]["redouble"] is None
    assert cond_ok([10, None], 12) and not cond_ok([10, 11], 12) and cond_ok(["a", "b"], "b")
    assert cond_ok(True, True) and not cond_ok(False, True)
    if os.path.exists(os.path.join(HERE, "rules.json")):
        rules = load_rules()
        for auc, seat, hand in (("1H P", 2, "KQ52.J84.A73.962"), ("1H P", 2, "AQ52.J84.A73.K62"),
                                ("P 1C", 2, "KQJ52.84.A73.962"), ("1S P 2S P", 0, "AKJ52.K84.A73.62")):
            out = advise(parse_calls(auc), 0, seat, parse_hand(hand), rules)
            assert out is not None, (auc, hand)
            print(f"  {auc:12s} {hand:18s} -> {out['call_name']:3s} [{out['rule']['id']}] {out['branch']['say']}")
    print("self-test ok")


if __name__ == "__main__":
    _selftest()
