"""Players: a cookie, a rating, and an optional name. The start of a user system.

There is no login. The first request from a browser sets the `bp_player` cookie
to a random id, and that id is the player. A player who picks a name shows on the
leaderboard; one who does not is still rated.

    GET  /api/me             the player's name, rating and champion flag
    POST /api/me/name        pick or change the name
    GET  /api/leaderboard    the named players, best first

The rating is Elo. Every board of a challenge the server dealt is one game against
the bots, who are fixed at 1500: a win when the user's table scores more IMPs, a
draw on zero. Boards from a challenge link are not rated, because the link carries
the other table's moves and anyone can write one. A board counts once per player,
keyed by its deal. A rated board that is started and left unfinished counts as a
loss, when that player next starts a challenge.

The data lives in SQLite at `$BRIDGE_DATA/players.db` (default `data/` beside the
code; deploy.sh leaves that folder alone on the server).
"""

import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from flask import g, jsonify, request

BOT_RATING = 1500
START_RATING = 1500
K = 24
CHAMPION_MIN_BOARDS = 8      # the champion's deck needs a record, not one lucky board
COOKIE = "bp_player"
NAME_RE = re.compile(r"^[A-Za-z0-9_\- ]{3,20}$")

DATA_DIR = Path(os.environ.get("BRIDGE_DATA", Path(__file__).resolve().parents[1] / "data"))
_LOCAL = threading.local()


def db():
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DATA_DIR / "players.db", timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players(
                id TEXT PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE,
                rating INTEGER NOT NULL, boards INTEGER NOT NULL DEFAULT 0,
                best INTEGER NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS boards(
                player TEXT NOT NULL, deal TEXT NOT NULL, state TEXT NOT NULL,
                imps INTEGER, before INTEGER, after INTEGER, at REAL NOT NULL,
                PRIMARY KEY(player, deal));
        """)
        _LOCAL.conn = conn
    return conn


# ------------------------------------------------------------------ who is asking

def current_id():
    """The player behind this request. A new one gets an id; the cookie is set after."""
    pid = request.cookies.get(COOKIE, "")
    if not re.fullmatch(r"[0-9a-f]{32}", pid):
        pid = getattr(g, "new_player", None) or secrets.token_hex(16)
        g.new_player = pid
    return pid


def ensure(pid):
    db().execute("INSERT OR IGNORE INTO players(id, rating, best, created) VALUES(?,?,?,?)",
                 (pid, START_RATING, START_RATING, time.time()))


def champion_id():
    row = db().execute("SELECT id FROM players WHERE name IS NOT NULL AND boards >= ? "
                       "ORDER BY rating DESC, boards DESC, created LIMIT 1",
                       (CHAMPION_MIN_BOARDS,)).fetchone()
    return row["id"] if row else None


def me(pid):
    row = db().execute("SELECT * FROM players WHERE id=?", (pid,)).fetchone()
    if row is None:
        return {"name": None, "rating": START_RATING, "boards": 0, "best": START_RATING,
                "champion": False, "bots": BOT_RATING}
    return {"name": row["name"], "rating": row["rating"], "boards": row["boards"],
            "best": row["best"], "champion": champion_id() == pid, "bots": BOT_RATING}


# ------------------------------------------------------------------ rating

def expected(rating):
    return 1 / (1 + 10 ** ((BOT_RATING - rating) / 400))


def _apply(conn, pid, deal, imps):
    """Rate one board inside a transaction. Returns (before, after)."""
    r = conn.execute("SELECT rating FROM players WHERE id=?", (pid,)).fetchone()["rating"]
    score = 1 if imps > 0 else 0 if imps < 0 else .5
    after = round(r + K * (score - expected(r)))
    conn.execute("UPDATE players SET rating=?, boards=boards+1, best=MAX(best, ?) WHERE id=?",
                 (after, after, pid))
    conn.execute("UPDATE boards SET state='done', imps=?, before=?, after=?, at=? "
                 "WHERE player=? AND deal=?", (imps, r, after, time.time(), pid, deal))
    return r, after


def board_started(pid, deal):
    """A rated board is on the table. False if this player already had it."""
    ensure(pid)
    cur = db().execute("INSERT OR IGNORE INTO boards(player, deal, state, at) VALUES(?,?,'open',?)",
                       (pid, deal, time.time()))
    return cur.rowcount == 1


def board_finished(pid, deal, imps):
    """Rate a finished board once. Returns {before, after} or None if it was not open."""
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT state FROM boards WHERE player=? AND deal=?", (pid, deal)).fetchone()
        if row is None or row["state"] != "open":
            conn.execute("COMMIT")
            return None
        before, after = _apply(conn, pid, deal, imps)
        conn.execute("COMMIT")
        return {"before": before, "after": after}
    except Exception:
        conn.execute("ROLLBACK")
        raise


def forfeit_open(pid):
    """Every rated board this player started and left counts as a loss now."""
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for row in conn.execute("SELECT deal FROM boards WHERE player=? AND state='open' ORDER BY at",
                                (pid,)).fetchall():
            _apply(conn, pid, row["deal"], -1)
            conn.execute("UPDATE boards SET state='forfeit' WHERE player=? AND deal=?", (pid, row["deal"]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def board_rating(pid, deal):
    row = db().execute("SELECT before, after FROM boards WHERE player=? AND deal=? AND state='done'",
                       (pid, deal)).fetchone()
    return {"before": row["before"], "after": row["after"]} if row else None


# ------------------------------------------------------------------ names and the board

def set_name(pid, name):
    """Returns an error string, or None."""
    name = " ".join((name or "").split())
    if not NAME_RE.match(name):
        return "a name is 3 to 20 letters, digits, spaces, - or _"
    ensure(pid)
    try:
        db().execute("UPDATE players SET name=? WHERE id=?", (name, pid))
    except sqlite3.IntegrityError:
        return "that name is taken"
    return None


def leaderboard(limit=50):
    champ = champion_id()
    rows = db().execute("SELECT id, name, rating, boards, best FROM players WHERE name IS NOT NULL "
                        "AND boards > 0 ORDER BY rating DESC, boards DESC, created LIMIT ?",
                        (limit,)).fetchall()
    return [{"name": r["name"], "rating": r["rating"], "boards": r["boards"], "best": r["best"],
             "champion": r["id"] == champ, "id": r["id"]} for r in rows]


# ------------------------------------------------------------------ routes

def register(app):
    @app.after_request
    def set_cookie(resp):
        pid = g.get("new_player")
        if pid:
            resp.set_cookie(COOKIE, pid, max_age=2 * 365 * 24 * 3600, httponly=True,
                            samesite="Lax", secure=request.is_secure)
        return resp

    @app.get("/api/me")
    def api_me():
        return jsonify(me(current_id()))

    @app.post("/api/me/name")
    def api_me_name():
        body = request.get_json(silent=True) or {}
        pid = current_id()
        err = set_name(pid, body.get("name"))
        if err:
            return jsonify(error=err), 400
        return jsonify(me(pid))

    @app.get("/api/leaderboard")
    def api_leaderboard():
        pid = current_id()
        rows = leaderboard()
        for r in rows:
            r["you"] = r.pop("id") == pid
        return jsonify(players=rows, me=me(pid), min_boards=CHAMPION_MIN_BOARDS)
