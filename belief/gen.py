"""multi-system auction generator for the belief net.

Plays every bidder in SYSTEMS against every other (and itself) on fresh deals from
dds_results_100M.npy and writes small shards. Hands are NOT stored: rebuild them from
``deal_index`` with ``load_range``. Cross pairs give two auctions per deal (A NS / B EW and
swapped); self pairs give one.

    OMP_NUM_THREADS=4 python -u belief/gen.py test     # fixed test set
    OMP_NUM_THREADS=4 python -u belief/gen.py train    # runs until stopped
    # round 2: half the new shards involve hi3_lo or hi52k
    OMP_NUM_THREADS=4 python -u belief/gen.py train --focus hi3_lo,hi52k --focus-frac 0.5

Shard rows (one per auction):
    deal_index int32, dealer int8, vul int8 (0 none, 1 NS, 2 EW, 3 both),
    ns_sys int8, ew_sys int8 (ids into SYSTEMS), hist int8 (n, T) padded with -1.

Deal ranges: train 0..90M (shard k uses [k*SHARD_DEALS, ...)), test 95M.. (round-1 pairs, ids
0..15, 120 x 1000 deals), test 96M.. (pairs involving an id >= 16, j-th such pair in all_pairs()
order gets [96M + j*1000, ...)), WBridge5 and match.py's eval tail (last 10k) are elsewhere.
Ids are append-only: new bidders go at the end of SYSTEMS.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import torch

# The bots that bid the auctions (match.py players, EPBot, brl) live in the lab repo.
ROOT = Path(__file__).resolve().parents[1]
LAB = Path(os.environ.get("BRIDGE_LAB", Path.home() / "code" / "bridge" / "lab"))
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(LAB / "experiments" / "match"))
sys.path.insert(0, str(LAB / "experiments" / "epbot"))
import match  # noqa: E402

KEEP = os.environ.get("BRIDGE_KEEP", str(Path.home() / "code" / "bridge" / "archive" / "notes" / "keep"))
OUT = Path(__file__).resolve().parent / "data"
SHARD_DEALS = 2000
TEST_FIRST, TEST_DEALS_PER_PAIR = 95_000_000, 1000
N_SYS_V1, TEST2_FIRST = 16, 96_000_000         # round-2 test pairs (any id >= 16)

# id -> (name, match.py spec). Id 0 is WBridge5 (dataset only, never generated here).
SYSTEMS = [
    ("wb5", None),
    ("brl_fsp", f"brl:{KEEP}/brl/brl_fsp_weights.npz"),
    ("brl_sl", f"brl:{KEEP}/brl/brl_sl_weights.npz"),
    ("D75", f"four:{KEEP}/bid/D75_pidgin-d-codeword_s75k.pt"),
    ("e2b", f"four:{KEEP}/bid/e2b_punisher-league-25pct_s24k.pt"),
    ("E46", f"four:{KEEP}/bid/E46_league-anyseat-table_s40k.pt"),
    ("E49B", f"four:{KEEP}/bid/E49B_grounded-to-table_s60k.pt"),
    ("E28", f"four:{KEEP}/bid/E28_xx-sac-passout_s50k.pt"),
    ("ep_21gf", "epbot:21gf"),
    ("ep_sayc", "epbot:sayc"),
    ("ep_acol", "epbot:acol"),
    ("ep_wj", "epbot:wj"),
    ("ep_prec", "epbot:pc"),
    ("ep_gib", "epbot:gib"),
    ("ep_wb5", "epbot:WBridge5-Sayc"),
    ("ep_ben", "epbot:BEN-SAYC"),
    # round 2 (2026-10-04)
    ("hi3_lo", f"four:{OUT.parent}/runs/tourney/models/hi3_lo_step30000.pt"),   # g_s2o_hi3_lo step 30k
    ("hi52k", f"four:{ROOT}/runs/box_pull/g_s2o_hi/3/ckpt_step52000.pt"),
]
GEN_IDS = [i for i, (_, s) in enumerate(SYSTEMS) if s]


class Pool:
    """Players built once; every EPBot player shares one process pool."""

    def __init__(self, workers: int):
        self.workers = workers
        self.shared = ProcessPoolExecutor(workers)
        self.players = {}

    def restart(self):
        """libEPBot segfaults now and then (inside epbot_set_system_type); start fresh workers."""
        self.shared.shutdown(wait=False, cancel_futures=True)
        self.shared = ProcessPoolExecutor(self.workers)
        for p in self.players.values():
            if hasattr(p, "pool"):
                p.pool = self.shared

    def get(self, i: int):
        if i not in self.players:
            p = match.make_player(SYSTEMS[i][1])
            if hasattr(p, "pool"):
                p.pool.shutdown()
                p.pool = self.shared
            self.players[i] = p
        return self.players[i]


def play_pair(pool: Pool, a: int, b: int, first: int, n: int, seed: int, tries: int = 3):
    for k in range(tries):
        try:
            return _play_pair(pool, a, b, first, n, seed)
        except BrokenProcessPool:
            print(f"EPBot worker died ({SYSTEMS[a][0]} v {SYSTEMS[b][0]}), retry {k + 1}", flush=True)
            pool.restart()
    return None


def _play_pair(pool: Pool, a: int, b: int, first: int, n: int, seed: int) -> dict:
    deals = match.load_range(match.DATA, first, n)
    g = torch.Generator().manual_seed(seed)
    deal = torch.arange(n)
    dealer = torch.randint(4, (n,), generator=g)
    vul = torch.randint(4, (n,), generator=g)
    if a == b:
        ctrl = torch.zeros(n, 2, dtype=torch.long)
        players = [pool.get(a)]
        sys_ns = np.full(n, a); sys_ew = np.full(n, a)
    else:
        deal, dealer, vul = deal.repeat(2), dealer.repeat(2), vul.repeat(2)
        ctrl = torch.cat([torch.tensor([[0, 1]]).expand(n, 2), torch.tensor([[1, 0]]).expand(n, 2)])
        players = [pool.get(a), pool.get(b)]
        sys_ns = np.r_[np.full(n, a), np.full(n, b)]; sys_ew = np.r_[np.full(n, b), np.full(n, a)]
    vns, vew = (vul == 1) | (vul == 3), (vul == 2) | (vul == 3)
    tab = match.play(players, deals, deal, dealer, vns, vew, ctrl)
    hist = match.trim_history(tab.history.numpy()).astype(np.int8)
    return dict(deal_index=(deal.numpy() + deals.first_index).astype(np.int32),
                dealer=dealer.numpy().astype(np.int8), vul=vul.numpy().astype(np.int8),
                ns_sys=sys_ns.astype(np.int8), ew_sys=sys_ew.astype(np.int8), hist=hist)


def all_pairs():
    return [(a, b) for a in GEN_IDS for b in GEN_IDS if a <= b]


def test_plan(first=()):
    """[(a, b, first deal, seed)] for every test pair. Round-1 pairs keep their old ranges/seeds;
    pairs with an id >= N_SYS_V1 get the 96M range. ``first``: system names whose pairs run first."""
    old = [(a, b) for a, b in all_pairs() if b < N_SYS_V1]
    new = [(a, b) for a, b in all_pairs() if b >= N_SYS_V1]
    plan = [(a, b, TEST_FIRST + k * TEST_DEALS_PER_PAIR, k) for k, (a, b) in enumerate(old)]
    plan += [(a, b, TEST2_FIRST + j * TEST_DEALS_PER_PAIR, 10_000 + j) for j, (a, b) in enumerate(new)]
    ids = {i for i, (n, _) in enumerate(SYSTEMS) if n in first}
    return sorted(plan, key=lambda x: not ({x[0], x[1]} <= ids))      # stable: priority pairs first


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "train"])
    ap.add_argument("workers", nargs="?", type=int, default=8)
    ap.add_argument("--focus", default="", help="train: comma list of system names to oversample")
    ap.add_argument("--focus-frac", type=float, default=0.5, help="train: share of shards with a --focus system")
    ap.add_argument("--minutes", type=float, default=0, help="train: stop after this many minutes (0 = never)")
    ap.add_argument("--first", default="", help="test: comma list of names; pairs among them run first")
    args = ap.parse_args()
    mode = args.mode
    torch.set_num_threads(4)
    pool = Pool(args.workers)
    (OUT / "systems.json").write_text(json.dumps([n for n, _ in SYSTEMS]))
    if mode == "test":
        d = OUT / "test"; d.mkdir(parents=True, exist_ok=True)
        for a, b, first, seed in test_plan(args.first.split(",")):
            f = d / f"pair_{a:02d}_{b:02d}.npz"
            if f.exists():
                continue
            t0 = time.time()
            r = play_pair(pool, a, b, first, TEST_DEALS_PER_PAIR, seed)
            if r is None:
                print(f"test {SYSTEMS[a][0]} v {SYSTEMS[b][0]}: SKIPPED", flush=True)
                continue
            tmp = d / f"pair_{a:02d}_{b:02d}.tmp.npz"
            np.savez_compressed(tmp, **r)
            tmp.rename(f)
            print(f"test {SYSTEMS[a][0]} v {SYSTEMS[b][0]}: {len(r['hist'])} auctions {time.time()-t0:.1f}s",
                  flush=True)
        return
    d = OUT / "train"; d.mkdir(parents=True, exist_ok=True)
    pairs = all_pairs()
    focus = {i for i, (n, _) in enumerate(SYSTEMS) if n in args.focus.split(",")}
    hot = [pr for pr in pairs if set(pr) & focus]
    rng = np.random.default_rng(int(time.time()))
    done = {int(p.stem.split("_")[1]) for p in d.glob("shard_*.npz")}
    k = max(done, default=-1) + 1
    t_start, n_auc = time.time(), 0
    while not args.minutes or time.time() - t_start < args.minutes * 60:
        pick = hot if hot and rng.random() < args.focus_frac else pairs
        a, b = pick[rng.integers(len(pick))]
        t0 = time.time()
        r = play_pair(pool, a, b, k * SHARD_DEALS, SHARD_DEALS, 1_000_000 + k)
        if r is None:
            k += 1
            continue
        tmp = d / f"shard_{k:06d}_{a:02d}_{b:02d}.tmp.npz"
        np.savez_compressed(tmp, **r)
        tmp.rename(d / f"shard_{k:06d}_{a:02d}_{b:02d}.npz")
        n_auc += len(r["hist"])
        print(f"shard {k} {SYSTEMS[a][0]} v {SYSTEMS[b][0]}: {len(r['hist'])} in {time.time()-t0:.1f}s "
              f"| total {n_auc} at {n_auc / (time.time()-t_start):.0f}/s", flush=True)
        k += 1
    pool.shared.shutdown(wait=False, cancel_futures=True)
    print(f"time limit: {n_auc} auctions in {(time.time() - t_start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
