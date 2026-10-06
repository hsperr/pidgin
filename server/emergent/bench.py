"""Bot benchmark: anyone's bidding bot gets the weak-spot report our own nets get.

- ``GET  /bench`` -- how it works, and the reports their owners chose to list
- ``GET  /apis/bench/deals`` -- the fixed deal set: JSON (``?offset=&limit=``) or
  ``?format=pbn``. Hands, dealer and vulnerability only; the DD tricks stay here.
- ``POST /apis/bench/report`` -- a duplicate match on those deals -> ``{"id", "url"}``
- ``GET  /bench/report/<id>`` -- the report page (``?format=json`` for the numbers)
- ``GET  /apis/bench/openapi.json`` and ``GET /apis/bench/agent.md`` -- the same contract as
  OpenAPI 3.1 and as a plain-text guide, for people and coding agents writing a client

A match: every board is played at two tables. At one your bot sits North-South, at
the other East-West, against the same opponent (any bot). The body is

    {"bot": "MyBot 1.2", "opponent": "GIB", "public": false,
     "boards": [{"board": 1, "ns": "1H P 1S P 2S P P P", "ew": "P 1D X P ..."}]}

``ns``: the auction where your bot sat N-S; ``ew``: where it sat E-W. Calls run from
the dealer, split by spaces, commas or dashes (or a JSON list): 1C..7NT, P, X, XX and
the usual spellings. Each auction must be legal and finished. Contracts are scored
double-dummy, so card play plays no part.

The numbers come from emergent/analysis/, frozen copies of bridge_public/tools/.
Reports are files in data/bench/, named by a hash of the body (the same match gives
the same id).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import threading
import time

import numpy as np
from flask import Response, jsonify, request

from training.bridge.auction import AuctionState
from emergent.analysis import simplicity, weakspots
from emergent.apis import ApiError, parse_call
from emergent.deck import RANKS, SUITS

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_FILE = os.path.join(HERE, "..", "models", "bench_100k.npz")
REPORT_DIR = os.environ.get("BENCH_REPORT_DIR", os.path.join(HERE, "..", "data", "bench"))
SET_NAME, SET_FIRST, SET_SIZE = "v1", 90_000, 10_000   # /play shows the first rows; not these
MIN_BOARDS, MAX_BODY = 200, 8 << 20
SEATS, VUL_NAMES = "NESW", ("None", "NS", "EW", "All")
MAX_CALLS = 320
DEALS: dict = {}
LOCK = threading.Lock()


def deals() -> dict:
    """The set: hands (n, 52) owners, tricks (n, 4, 5), dealer, vul code (0 none .. 3 all)."""
    if not DEALS:
        z = np.load(BENCH_FILE)
        rows = slice(SET_FIRST, SET_FIRST + SET_SIZE)
        DEALS.update(owners=z["hands"][rows].argmax(1).astype(np.int64),
                     tricks=z["tricks"][rows].astype(np.int64),
                     dealer=z["dealer"][rows].astype(np.int64),
                     vul=(z["vul_ns"][rows].astype(int) + 2 * z["vul_ew"][rows].astype(int)),
                     deal_first=int(z["deal_first"]) + SET_FIRST)
    return DEALS


def hand_text(owners: np.ndarray, seat: int) -> str:
    return ".".join("".join(RANKS[c % 13] for c in range(s * 13, s * 13 + 13) if owners[c] == seat)
                    for s in range(4))


def board_json(i: int) -> dict:
    d = deals()
    return {"board": i + 1, "dealer": SEATS[d["dealer"][i]], "vul": VUL_NAMES[d["vul"][i]],
            "hands": {SEATS[s]: hand_text(d["owners"][i], s) for s in range(4)}}


def pbn(first: int, stop: int) -> str:
    d, out = deals(), []
    for i in range(first, stop):
        hands = " ".join(hand_text(d["owners"][i], s) for s in range(4))
        out.append(f'[Event "training bench {SET_NAME}"]\n[Board "{i + 1}"]\n'
                   f'[Dealer "{SEATS[d["dealer"][i]]}"]\n[Vulnerable "{VUL_NAMES[d["vul"][i]]}"]\n'
                   f'[Deal "N:{hands}"]\n')
    return "\n".join(out)


# ------------------------------------------------------------------ scoring

def parse_auction(raw, board: int, dealer: int, vul: int, table: str) -> AuctionState:
    tokens = raw if isinstance(raw, list) else re.split(r"[\s,\-]+", str(raw or "").strip())
    try:
        calls = [parse_call(str(t)) for t in tokens if str(t)]
        state = AuctionState.from_calls(calls, dealer, vul in (1, 3), vul in (2, 3))
    except (ApiError, ValueError) as e:
        raise ApiError(f"board {board} {table}: {e}") from None
    if not state.ended:
        raise ApiError(f"board {board} {table}: the auction has not ended")
    return state


def score_match(body: dict) -> dict:
    """Body -> match.py-style arrays (A = the submitted bot), checked and scored DD."""
    rows = body.get("boards")
    if not isinstance(rows, list) or not MIN_BOARDS <= len(rows) <= SET_SIZE:
        raise ApiError(f"'boards' must list {MIN_BOARDS} to {SET_SIZE} boards")
    d = deals()
    idx = [int(r.get("board", 0)) - 1 if isinstance(r, dict) else -1 for r in rows]
    if min(idx) < 0 or max(idx) >= SET_SIZE:
        raise ApiError(f"every board needs a 'board' number from 1 to {SET_SIZE}")
    if len(set(idx)) != len(idx):
        raise ApiError("a board is listed twice")
    n = len(rows)
    z = {k: np.zeros(n, np.int64) for k in ("c1", "c2", "d1", "d2", "calls1", "calls2")}
    z.update(hist1=np.full((n, MAX_CALLS), -1, np.int64), hist2=np.full((n, MAX_CALLS), -1, np.int64),
             dealer=d["dealer"][idx], vul=d["vul"][idx], deal_index=np.array(idx) + d["deal_first"])
    doubled = np.zeros((2, n), np.int64)
    for k, (i, row) in enumerate(zip(idx, rows)):
        for t, key in ((1, "ns"), (2, "ew")):
            state = parse_auction(row.get(key), i + 1, int(d["dealer"][i]), int(d["vul"][i]), key)
            z[f"hist{t}"][k, :len(state.calls)] = state.calls
            z[f"calls{t}"][k] = len(state.calls)
            z[f"c{t}"][k], z[f"d{t}"][k] = state.last_contract, state.declarer()
            doubled[t - 1, k] = state.doubled
    tricks = d["tricks"][idx]
    vul_ns, vul_ew = np.isin(z["vul"], (1, 3)), np.isin(z["vul"], (2, 3))
    for t in (1, 2):
        z[f"ns{t}"], _ = weakspots.table_score(tricks, vul_ns, vul_ew, z[f"c{t}"], z[f"d{t}"],
                                               doubled[t - 1])
    z["imps_A"] = weakspots.imps(z["ns1"] - z["ns2"])
    width = max(1, int(max(z["calls1"].max(), z["calls2"].max())))
    z["hist1"], z["hist2"] = z["hist1"][:, :width], z["hist2"][:, :width]
    return {"z": z, "owners": d["owners"][idx], "tricks": tricks}


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000) -> list[float]:
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def build_report(body: dict) -> dict:
    m = score_match(body)
    z, owners = m["z"], m["owners"]
    imps = z["imps_A"].astype(float)
    return {
        "set": SET_NAME, "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "bot": str(body.get("bot") or "your bot")[:60],
        "opponent": str(body.get("opponent") or "opponent")[:60],
        "public": bool(body.get("public", False)),
        "imps_ci95": bootstrap_ci(imps),
        "won_lost_tied": [int((imps > 0).sum()), int((imps < 0).sum()), int((imps == 0).sum())],
        "weak": weakspots.analyse(z, owners, m["tricks"]),
        "openings": {w: weakspots.openings(z, owners, w) for w in ("A", "B")},
        "style": {w: simplicity.analyse(z, owners=owners, who=w) for w in ("A", "B")},
    }


def report_path(rid: str) -> str:
    return os.path.join(REPORT_DIR, f"{rid}.json")


def listed_reports(limit: int = 30) -> list[dict]:
    if not os.path.isdir(REPORT_DIR):
        return []
    files = sorted((f for f in os.listdir(REPORT_DIR) if f.endswith(".json")),
                   key=lambda f: os.path.getmtime(os.path.join(REPORT_DIR, f)), reverse=True)
    out = []
    for f in files:
        with open(os.path.join(REPORT_DIR, f)) as fh:
            r = json.load(fh)
        if r.get("public"):
            out.append({"id": f[:-5], "bot": r["bot"], "opponent": r["opponent"], "when": r["when"],
                        "boards": r["weak"]["boards"], "imps": r["weak"]["imps_per_board"]})
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ pages

CSS = """
:root{--bg:#f6f5f1;--card:#fff;--line:#d9d5cb;--text:#1d1f23;--dim:#5f636b;--faint:#9a9da4;
--accent:#2d5bd8;--pos:#1f8a5b;--neg:#c24a3a;--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121417;--card:#1b1e23;--line:#343943;--text:#e8eaed;
--dim:#a3a8b1;--faint:#6b707a;--accent:#7598ff;--pos:#4cc08a;--neg:#ef7866}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;padding:24px 16px 64px}
.wrap{max-width:980px;margin:0 auto}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 6px}
.dim{color:var(--dim)}.small{font-size:13px}a{color:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.big{font-size:34px;font-weight:650;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px;margin:6px 0}
th,td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:right;white-space:nowrap}
th{color:var(--dim);font-weight:600}th:first-child,td:first-child{text-align:left}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;
font:13px/1.45 var(--mono)}code{font-family:var(--mono);font-size:.92em}
details summary{cursor:pointer;color:var(--dim);margin:8px 0}
"""


def page(title: str, body: str) -> Response:
    return Response(f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<title>{html.escape(title)}</title><style>{CSS}</style></head>'
                    f'<body><div class="wrap">{body}</div></body></html>',
                    mimetype="text/html")


def sgn(v, digits=2):
    return "–" if v is None else f"{v:+.{digits}f}"


def cls(v, good=1):
    return "" if v is None or v == 0 else "pos" if v * good > 0 else "neg"


def table(head: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(c if str(c).startswith("<td") else f"<td>{c}</td>" for c in r)
                   + "</tr>" for r in rows)
    return f'<div class="scroll"><table><tr>{th}</tr>{body}</table></div>'


REPORT_PAGE = os.path.join(HERE, "bench_report.html")


def report_html(r: dict) -> Response:
    """The report page: one template that draws itself from the report's JSON."""
    with open(REPORT_PAGE) as fh:
        template = fh.read()
    data = json.dumps(r).replace("</", "<\\/")          # no early </script> from a bot's name
    title = html.escape(f"{r['bot']} vs {r['opponent']}")
    return Response(template.replace("__TITLE__", title).replace("/*REPORT*/", data),
                    mimetype="text/html")


def index_html() -> Response:
    listed = listed_reports()
    rows = [[f'<a href="/bench/report/{x["id"]}">{html.escape(x["bot"])}</a>', html.escape(x["opponent"]),
             x["boards"], f'<td class="{cls(x["imps"])}">{sgn(x["imps"])}</td>', x["when"]] for x in listed]
    reports = (table(["Bot", "Opponent", "Boards", "IMPs/board", "When"], rows) if rows
               else '<p class="dim">No listed reports yet.</p>')
    host = request.host_url.rstrip("/")
    body = f"""
<h1>Bot benchmark</h1>
<p class="dim">Test your bidding bot on a fixed set of {SET_SIZE:,} deals and get a report: where it
wins and loses IMPs, how it competes and doubles, what its openings show, how natural its bidding is.
The same report we use for our own nets.</p>
<h2>1. Get the deals</h2>
<pre>curl '{host}/apis/bench/deals?offset=0&amp;limit=1000'   # JSON, up to 1000 per call
curl '{host}/apis/bench/deals?format=pbn' &gt; bench_{SET_NAME}.pbn</pre>
<p class="dim small">Hands are S.H.D.C. Each deal has its own dealer and vulnerability.</p>
<h2>2. Play a duplicate match</h2>
<p class="dim">Play every board twice against the same opponent (any bot): once with your bot North-South,
once East-West. Only the auction matters; the contract is scored double-dummy. Use at least
{MIN_BOARDS} boards; 2,000 or more give a clear result.</p>
<h2>3. Send the auctions</h2>
<pre>curl -X POST {host}/apis/bench/report -H 'Content-Type: application/json' -d '{{
  "bot": "MyBot 1.2", "opponent": "GIB", "public": false,
  "boards": [
    {{"board": 1, "ns": "1H P 1S P 2S P P P", "ew": "P 1D X P 1S P P P"}},
    ...
  ]}}'
# -> {{"id": "3f2a...", "url": "{host}/bench/report/3f2a..."}}</pre>
<p class="dim small"><code>ns</code> is the auction where your bot sat N-S, <code>ew</code> where it sat E-W.
Calls start with the dealer: 1C…7NT, P, X, XX. <code>"public": true</code> lists the report below;
otherwise only the link finds it.</p>
<p class="dim small">Writing a client, or asking a coding agent to? The whole contract:
<a href="/apis/bench/agent.md">agent.md</a> (plain text) ·
<a href="/apis/bench/openapi.json">openapi.json</a>.</p>
<h2>Listed reports</h2>
{reports}"""
    return page("Bot benchmark", body)


AGENT_MD = """# Bot benchmark API ({host})

Goal: get a weak-spot report for a bridge bidding bot. Three steps, no auth.

## 1. Get the deals

    GET {host}/apis/bench/deals?offset=0&limit=1000

At most 1000 boards per call; the set has {size} boards (board numbers 1..{size}). Answer:

    {{"set": "{set}", "total": {size}, "offset": 0,
     "boards": [{{"board": 1, "dealer": "E", "vul": "NS",
                 "hands": {{"N": "AK52.T4.QJ3.K862", "E": "...", "S": "...", "W": "..."}}}}]}}

- hands: suits in the order spades.hearts.diamonds.clubs, ranks AKQJT98765432, a void is empty.
- dealer: N, E, S or W. vul: None, NS, EW or All. Both differ per board: use them.
- The same deals come back every time. `?format=pbn` gives them all as one PBN file.

## 2. Play a duplicate match

For every board you use, run two auctions against the same opponent (any bot):
- `ns`: your bot sits North and South, the opponent East and West.
- `ew`: your bot sits East and West, the opponent North and South.

Each seat must see only its own hand and the calls so far. Calls start with the
dealer and go clockwise N, E, S, W. The auction ends after three passes following a
bid, or four passes at the start. Only the auction is needed: the final contract is
scored double-dummy on the server.

## 3. Post the auctions

    POST {host}/apis/bench/report
    Content-Type: application/json

    {{"bot": "MyBot 1.2", "opponent": "GIB", "public": false,
     "boards": [{{"board": 1, "ns": "1H P 1S P 2S P P P", "ew": "P 1D X P 1S P P P"}}]}}

- boards: {min_boards} to {size} entries, each board number at most once.
- Calls: 1C..7C, 1D.., 1H.., 1S.., 1NT..7NT (also 1N), P (Pass), X (Double), XX (Redouble).
  Separate calls by spaces, commas or dashes, or send a JSON list of strings.
- Every auction must be legal and finished, or the whole post is refused.
- public: true lists the report on {host}/bench; false (default) keeps it link-only.
- Body limit: {max_mb} MB (10000 boards is about 1 MB).

Answer 200:

    {{"id": "3f2a9c...", "url": "{host}/bench/report/3f2a9c...", "imps_per_board": -0.21}}

Answer 400: {{"error": "board 17 ew: illegal call ..."}}. Fix that board and post again.
The same body always gives the same id.

## 4. Read the report

- {host}/bench/report/<id> : the HTML page
- {host}/bench/report/<id>?format=json : all numbers

JSON keys: `weak.imps_per_board`, `imps_ci95`, `weak.contested` (IMPs by contested or not),
`weak.A` / `weak.B` (A = your bot, B = the opponent: doubled-and-down per 1000 tables,
punish_rate, doubles per level, outbid value by HCP), `weak.B1`, `weak.C`,
`openings.A` / `openings.B` (per opening: count, HCP, lengths; open rate by HCP as dealer),
`style.A` / `style.B` (code words and other style rates).

## Tips

- Use 2000 boards or more. With a few hundred boards the 95% range is wide.
- Machine-readable contract: {host}/apis/bench/openapi.json
"""


def openapi(host: str) -> dict:
    board = {"type": "object", "required": ["board", "dealer", "vul", "hands"], "properties": {
        "board": {"type": "integer", "minimum": 1, "maximum": SET_SIZE},
        "dealer": {"enum": list(SEATS)}, "vul": {"enum": list(VUL_NAMES)},
        "hands": {"type": "object", "description": "seat -> S.H.D.C, e.g. AK52.T4.QJ3.K862",
                  "properties": {s: {"type": "string"} for s in SEATS}}}}
    auction = {"oneOf": [{"type": "string", "example": "1H P 1S P 2S P P P"},
                         {"type": "array", "items": {"type": "string"}}],
               "description": "calls from the dealer: 1C..7NT, P, X, XX; legal and finished"}
    error = {"description": "refused", "content": {"application/json": {"schema": {
        "type": "object", "properties": {"error": {"type": "string"}}}}}}
    return {
        "openapi": "3.1.0",
        "info": {"title": "Bridge bot benchmark", "version": SET_NAME,
                 "description": f"Plain-text guide: {host}/apis/bench/agent.md"},
        "servers": [{"url": host}],
        "paths": {
            "/apis/bench/deals": {"get": {
                "summary": "The fixed deal set (no double-dummy tricks)",
                "parameters": [
                    {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 1000,
                                                                 "maximum": 1000}},
                    {"name": "format", "in": "query", "schema": {"enum": ["json", "pbn"]}}],
                "responses": {"200": {"description": "boards", "content": {
                    "application/json": {"schema": {"type": "object", "properties": {
                        "set": {"type": "string"}, "total": {"type": "integer"},
                        "offset": {"type": "integer"}, "boards": {"type": "array", "items": board}}}},
                    "text/plain": {"schema": {"type": "string", "description": "PBN"}}}}}}},
            "/apis/bench/report": {"post": {
                "summary": "Score a duplicate match and store its report",
                "requestBody": {"required": True, "content": {"application/json": {"schema": {
                    "type": "object", "required": ["boards"], "properties": {
                        "bot": {"type": "string", "maxLength": 60},
                        "opponent": {"type": "string", "maxLength": 60},
                        "public": {"type": "boolean", "default": False},
                        "boards": {"type": "array", "minItems": MIN_BOARDS, "maxItems": SET_SIZE,
                                   "items": {"type": "object", "required": ["board", "ns", "ew"],
                                             "properties": {"board": {"type": "integer"},
                                                            "ns": auction, "ew": auction}}}}}}}},
                "responses": {"200": {"description": "stored", "content": {"application/json": {
                    "schema": {"type": "object", "properties": {
                        "id": {"type": "string"}, "url": {"type": "string"},
                        "imps_per_board": {"type": "number"}}}}}},
                    "400": error, "413": error}}},
            "/bench/report/{id}": {"get": {
                "summary": "The report: HTML, or JSON with format=json",
                "parameters": [{"name": "id", "in": "path", "required": True,
                                "schema": {"type": "string", "pattern": "^[0-9a-f]{16}$"}},
                               {"name": "format", "in": "query", "schema": {"enum": ["json"]}}],
                "responses": {"200": {"description": "report"}, "404": {"description": "unknown id"}}}},
        },
    }


def register(app):
    @app.get("/bench")
    def bench_index():
        return index_html()

    @app.get("/apis/bench/openapi.json")
    def bench_openapi():
        return jsonify(openapi(request.host_url.rstrip("/")))

    @app.get("/apis/bench/agent.md")
    def bench_agent_md():
        text = AGENT_MD.format(host=request.host_url.rstrip("/"), size=SET_SIZE, set=SET_NAME,
                               min_boards=MIN_BOARDS, max_mb=MAX_BODY >> 20)
        return Response(text, mimetype="text/markdown")

    @app.get("/apis/bench/deals")
    def bench_deals():
        try:
            first = max(0, int(request.args.get("offset", 0)))
            limit = int(request.args.get("limit", SET_SIZE if request.args.get("format") == "pbn" else 1000))
        except ValueError:
            return jsonify(error="offset and limit must be numbers"), 400
        stop = min(SET_SIZE, first + max(0, limit))
        if request.args.get("format") == "pbn":
            return Response(pbn(first, stop), mimetype="text/plain",
                            headers={"Content-Disposition": f"attachment; filename=bench_{SET_NAME}.pbn"})
        return jsonify(set=SET_NAME, total=SET_SIZE, offset=first,
                       boards=[board_json(i) for i in range(first, min(stop, first + 1000))])

    @app.post("/apis/bench/report")
    def bench_report():
        if (request.content_length or 0) > MAX_BODY:
            return jsonify(error=f"body over {MAX_BODY >> 20} MB"), 413
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="send a JSON object with 'boards'"), 400
        rid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
        try:
            with LOCK:                     # one analysis at a time: a few CPU seconds each
                report = build_report(body)
        except ApiError as e:
            return jsonify(error=str(e)), 400
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(report_path(rid), "w") as fh:
            json.dump(report, fh)
        return jsonify(id=rid, url=f"{request.host_url.rstrip('/')}/bench/report/{rid}",
                       imps_per_board=report["weak"]["imps_per_board"])

    @app.get("/bench/report/<rid>")
    def bench_report_page(rid):
        if not re.fullmatch(r"[0-9a-f]{16}", rid) or not os.path.exists(report_path(rid)):
            return page("Not found", "<h1>No such report</h1>"), 404
        with open(report_path(rid)) as fh:
            report = json.load(fh)
        if request.args.get("format") == "json":
            return jsonify(report)
        return report_html(report)
