"""Opening-lead accuracy of the card-play models, double dummy.

DDOLAR: share of opening leads that give up no trick against the best lead double dummy
(the lead's DD tricks for the defence equal the max over all legal leads). ADDOLAR: the
same, skipping deals where every legal lead gives the same DD result. Definitions from
Hammond, detectingcheatinginbridge.com.

Boards: the first --boards contracts of server/models/bench_100k.npz (passed-out deals
dropped). Every legal lead is solved once and cached in --cache, which all models reuse.

    python scripts/opening_leads.py --boards 5000 --search 1000

The served players do not search the opening lead (CONFIG.defence_from = 2), so their
lead is the net's greedy card. --search N also measures the lead with PIMC forced on
(defence "lead", 20 layouts, seeded, no clock) on the first N boards.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
os.environ.setdefault("PLAY_SEARCH", "0")

from emergent import engine, playdesk, playq  # noqa: E402
from training.bridge.play import PlayBatch  # noqa: E402
from training.play.data import load_contracts  # noqa: E402
from training.play.model import encode  # noqa: E402
from training.play.search import make_deals, solved_values  # noqa: E402

MODELS = ROOT / "server" / "models"
BENCH = MODELS / "bench_100k.npz"
POLICY = "play_E48_wideleagueH.pt"
NAMES = {"policy": "policy net", "earlier": "earlier policy net", "qnet": "Q-net"}
FILES = {"policy": POLICY, "earlier": "play_E48_leagueE.pt", "qnet": "play_B2g_s540k.pt"}


def contracts(n):
    """The first `n` contracts of the bench. Rows of 120k raw deals is far more than enough."""
    c = load_contracts(BENCH, int(n * 1.3) + 100)
    if len(c) < n:
        raise SystemExit(f"only {len(c)} contracts in the bench")
    return c.subset(torch.arange(n))


def lead_table(c, cache):
    """(n, 52) int8: DD tricks for the defence after each legal lead, -1 where illegal."""
    n = len(c)
    if cache.exists():
        z = np.load(cache)
        if len(z["tricks"]) >= n:
            return z["tricks"][:n]
    t0 = time.time()
    out = np.full((n, 52), -1, dtype=np.int8)
    deals = make_deals(c)
    for start in range(0, n, 512):
        for i, vals in enumerate(solved_values(deals[start:start + 512]), start):
            for card, tricks in vals.items():
                out[i, card] = tricks
        print(f"  solved {min(start + 512, n)}/{n} leads tables, {time.time() - t0:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, tricks=out)
    return out


def load(key):
    if key == "qnet":
        sampler = playdesk.PlayBot(str(MODELS / POLICY))
        return playq.QPlayBot(str(MODELS / FILES[key]), sampler.net)
    return playdesk.PlayBot(str(MODELS / FILES[key]))


@torch.no_grad()
def greedy_leads(bot, c):
    batch = PlayBatch(c.owner.clone(), c.trump, c.declarer)
    if getattr(bot, "family", None) == "playq":
        return bot.values(c, batch).argmax(1).numpy()
    auction = bot.net.auction(c.calls, c.n_calls, c.dealer)
    enc = encode(batch, c, batch.t > 0, auction)
    return bot.net(enc["features"], batch.legal())["log_probs"].argmax(1).numpy()


def searched_leads(bot, c, n):
    """The lead engine.choose_card makes with PIMC forced on the opening lead, seeded."""
    engine.CONFIG = dataclasses.replace(engine.CONFIG, defence="lead")
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ci = c.subset(torch.tensor([i]))
        batch = PlayBatch(ci.owner.clone(), ci.trump, ci.declarer)
        out[i] = engine.choose_card(bot, ci, batch, search=True, seed=i)[0]
    return out


def score(tricks, leads):
    n = len(leads)
    got = tricks[np.arange(n), leads].astype(int)
    assert (got >= 0).all(), "an illegal lead"
    legal = tricks >= 0
    best = np.where(legal, tricks, -1).max(1)
    worst = np.where(legal, tricks, 99).min(1)
    ok = got == best
    varied = best != worst

    def rate(x):
        p = x.mean()
        return p, np.sqrt(p * (1 - p) / len(x)), len(x)

    return rate(ok), rate(ok[varied]), float((best - got).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=5000, help="boards for the greedy leads")
    ap.add_argument("--models", default="policy,earlier,qnet", help="any of policy, earlier, qnet")
    ap.add_argument("--search", type=int, default=0, help="boards for the forced-search leads (0 = skip)")
    ap.add_argument("--cache", type=Path, default=ROOT / "results" / "opening_leads_dd.npz")
    ap.add_argument("--out", type=Path, default=None, help="also write the table here")
    a = ap.parse_args()

    n = max(a.boards, a.search)
    c = contracts(n)
    tricks = lead_table(c, a.cache)
    lines = [f"Opening leads, first {n} contracts of {BENCH.name}. +/- is one standard error.",
             f"{'model':<34} {'DDOLAR':>15} {'ADDOLAR':>15} {'N':>6} {'N adj':>6} {'tricks lost':>11}"]

    def row(label, leads):
        (p, se, k), (q, sq, m), lost = score(tricks[:len(leads)], leads)
        lines.append(f"{label:<34} {100 * p:6.1f} +/- {100 * se:3.1f} {100 * q:6.1f} +/- {100 * sq:3.1f} "
                     f"{k:6d} {m:6d} {lost:11.3f}")
        print(lines[-1], flush=True)

    print("\n".join(lines), flush=True)
    for key in a.models.split(","):
        bot = load(key)
        row(f"{NAMES[key]}, greedy", greedy_leads(bot, c.subset(torch.arange(a.boards))))
        if a.search:
            t0 = time.time()
            leads = searched_leads(bot, c, a.search)
            row(f"{NAMES[key]}, PIMC on the lead", leads)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
