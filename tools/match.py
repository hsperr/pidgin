"""Duplicate teams match between two bidding models, scored in IMPs.

Every board is played at two tables on the same deal, dealer, and vulnerability:
table 1 seats player A North-South and player B East-West; table 2 swaps them.
All four seats bid. The Table supports Double and Redouble (legality and doubled/
redoubled scoring), but a player only doubles if it emits X/XX: phase1, zero,
D5OWN4 ``four`` players never do; D5OWN4X ``four`` players and the robot can.
The contract is the last bid; declarer is the first player of the winning side to
name its strain; the score uses that side's vulnerability and the final X/XX state.
A's IMPs on a board = imps(NS score at table 1 - NS score at table 2).

Players (``--a`` / ``--b``):

- ``pass`` -- always Passes; against it a model plays a silent-opponent auction,
  which must reproduce that model's saved silent eval scores (adapter check).
- ``four:PATH`` -- a four-seat D5OWN4 or D5OWN4X checkpoint (bridgezero.fourseat);
  greedy policy that also sees LHO/RHO bids, fed from the table's full call history.
  A D5OWN4X* net also sees the standing-doubled bits and may Double (never Redouble);
  a D5OWN4XC net also sees Pass-would-end-auction and standing-redoubled bits and may
  X/XX (and sacrifice) only where its Pass would end the auction.
  ``four:PATH:q>M`` (double-value nets): Double iff legal and double_value > M (/100),
  ignoring the X logit; quote the spec in a shell.
- ``zero:PATH[:RULE]`` -- a bridgezero auction checkpoint (D1/D2/D4 kind),
  RULE ``policy`` (default) or ``q``. Its feature vector only has slots for its
  own partnership's calls, so it does not see opponent bids; legality always
  comes from the real auction. Its ``expected`` rule assumes Pass keeps the
  partnership's own contract, which is false with active opponents, so it is
  refused.

Boards: the last ``--deals`` deals x 4 dealers x 4 vulnerabilities (none, NS,
EW, both); ``--boards N`` plays a seeded random subset of them. Checks: a
player against itself must score exactly 0 IMPs (skipped for the robot), and a
sample of auctions is replayed through bridgezero's ``AuctionState`` scoring.

    OMP_NUM_THREADS=8 python -u tools/match.py \
      --a four:runs/adversarial/ckpt_step60000.pt \
      --b four:runs/cooperative_only/ckpt_step20000.pt \
      --data data/deals.npz --out results/adversarial_vs_cooperative
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgezero.bridge.auction import AuctionState  # noqa: E402
from bridgezero.bridge.calls import PASS  # noqa: E402
from bridgezero.bridge.scoring import contract_score, terminal_ns_score  # noqa: E402
from bridgezero.contract.data import load_range  # noqa: E402
from bridgezero.contract.evaluate import imps_array, paired_bootstrap  # noqa: E402
from bridgezero.contract.model import load_checkpoint  # noqa: E402
from bridgezero.contract.prefixes import MAX_DECISIONS, CoopBatch, observe  # noqa: E402
from bridgezero.contract.targets import CONTRACT_TABLE_STRAIN, SCORE_LOOKUP  # noqa: E402
from bridgezero.fourseat.model import load_fourseat_checkpoint, policy_log_probs  # noqa: E402
from bridgezero.fourseat.competitive import MAX_REDOUBLE_CALLS, competitive_features  # noqa: E402
from bridgezero.fourseat.state import features_from_history  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = "data/deals.npz"
L = 35
# With X and XX (robot anywhere, D5OWN4XC nets at the pass-out seat) a legal auction can run
# 3 opening passes + 35 bids + 34 x "P P X P P XX P P" + a final "P P X P P XX P P P" = 319.
MAX_TABLE_CALLS = MAX_REDOUBLE_CALLS
DOUBLE, REDOUBLE = PASS + 1, PASS + 2
# DOUBLED_LOOKUP[vulnerable, doubled 0/1/2, contract, tricks] = declarer score.
DOUBLED_LOOKUP = torch.tensor([[[[contract_score(c // 5 + 1, c % 5, k, x, bool(v)) for k in range(14)]
                                 for c in range(L)] for x in range(3)] for v in range(2)],
                              dtype=torch.float32)
VUL_NAMES = ("none", "NS", "EW", "both")


import table_state as P1  # noqa: E402


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ table

class Table:
    """A batch of four-seat auctions in phase-1 ``St`` form plus per-side views."""

    def __init__(self, deals, deal, dealer, vul_ns, vul_ew, controller):
        n = len(deal)
        self.deals, self.deal, self.dealer = deals, deal, dealer
        self.vul_ns, self.vul_ew = vul_ns, vul_ew
        self.controller = controller                     # (n, 2) player index per side
        self.st = P1.St.empty(deal, L)
        self.t = 0
        # Only the robot can double; models never see X/XX in their inputs.
        self.doubled = torch.zeros(n, dtype=torch.long)          # 0, 1 = X, 2 = XX
        self.contract_side = torch.full((n,), -1, dtype=torch.long)
        self.bidder = torch.full((n, 2, L), -1, dtype=torch.long)
        self.k = torch.zeros(n, 2, dtype=torch.long)
        self.first_pass = torch.zeros(n, 2, dtype=torch.bool)
        self.history = torch.full((n, MAX_TABLE_CALLS), -1, dtype=torch.long)

    def seat_abs(self, rows: torch.Tensor) -> torch.Tensor:
        return (self.dealer[rows] + self.t) % 4

    def apply(self, action: torch.Tensor) -> None:
        alive = self.st.alive.nonzero().squeeze(1)
        a = action[alive]
        side = self.seat_abs(alive) % 2
        last, state, owner = self.st.last[alive], self.doubled[alive], self.contract_side[alive]
        dbl, rdbl = a == DOUBLE, a == REDOUBLE
        legal = ((a == PASS) | ((a < L) & (a > last))
                 | (dbl & (last >= 0) & (state == 0) & (side != owner))
                 | (rdbl & (state == 1) & (side == owner)))
        if not bool(legal.all()):
            raise AssertionError("illegal call")
        kk = self.k[alive, side]
        bid = a < PASS
        self.bidder[alive[bid], side[bid], a[bid]] = kk[bid] % 2
        self.doubled[alive[bid]] = 0
        self.contract_side[alive[bid]] = side[bid]
        self.doubled[alive[dbl]] = 1
        self.doubled[alive[rdbl]] = 2
        opening_pass = (a == PASS) & (kk == 0)
        self.first_pass[alive[opening_pass], side[opening_pass]] = True
        self.k[alive, side] += 1
        self.history[alive, self.t] = a
        # phase-1 St has no doubles: an X/XX row skips the update, then resets the pass count.
        x_rows = alive[dbl | rdbl]
        if len(x_rows):
            self.st.alive[x_rows] = False
        P1.apply_call_(self.st, action.clamp(max=PASS), self.t % 4, L)
        if len(x_rows):
            self.st.alive[x_rows] = True
            self.st.npass[x_rows] = 0
        self.t += 1

    def ns_scores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        st = self.st
        has = st.last >= 0
        c = st.last.clamp(min=0)
        decl = (P1.declarer_of(st, torch.tensor(P1.contract_strains()), L) + self.dealer) % 4
        vul = torch.where(decl % 2 == 0, self.vul_ns, self.vul_ew).long()
        tricks = self.deals.tricks[self.deal, decl, torch.as_tensor(CONTRACT_TABLE_STRAIN)[c]]
        raw = DOUBLED_LOOKUP[vul, self.doubled, c, tricks.long()]
        ns = torch.where(decl % 2 == 0, raw, -raw)
        ns = torch.where(has, ns, torch.zeros_like(ns))
        return ns.numpy(), st.last.numpy(), torch.where(has, decl, -1).numpy()


# ---------------------------------------------------------------- players

class ZeroPlayer:
    def __init__(self, path: str, rule: str = "policy"):
        if rule not in ("policy", "q"):
            raise ValueError("zero players support rule policy or q (expected assumes silent opponents)")
        self.net, ck = load_checkpoint(path, "cpu")
        self.net.eval()
        self.rule = rule
        self.observation = ck.get("observation", "intact")
        self.name = f"{Path(path).parent.name}:{rule}"
        self.meta = {"kind": "zero", "path": path, "sha256": sha256(path), "rule": rule,
                     "stage": ck.get("stage"), "step": ck.get("step"),
                     "observation": self.observation}

    @torch.no_grad()
    def act(self, table: Table, rows: torch.Tensor) -> torch.Tensor:
        seat = table.seat_abs(rows)
        side = seat % 2
        n = len(rows)
        batch = CoopBatch(
            deal=table.deal[rows], side=side, dealer=table.dealer[rows],
            vul=torch.where(side == 0, table.vul_ns[rows], table.vul_ew[rows]).long(),
            bidder=table.bidder[rows, side], first_pass=table.first_pass[rows, side],
            last=table.st.last[rows], k=table.k[rows, side],
            ended=torch.zeros(n, dtype=torch.bool),
            history=torch.full((n, MAX_DECISIONS), -1, dtype=torch.long))
        if not torch.equal(batch.actor_seat, seat):
            raise AssertionError("partnership view disagrees with the table seat")
        out = self.net(table.deals.hands[table.deal[rows], seat], observe(batch, self.observation))
        values = out["policy_logits"] if self.rule == "policy" else out["contract_q"]
        legal = torch.cat([torch.arange(L)[None] > table.st.last[rows, None],
                           torch.ones(n, 1, dtype=torch.bool)], 1)
        return values[:, :L + 1].masked_fill(~legal, -torch.inf).argmax(-1)


class FourSeatPlayer:
    """Four-seat D5OWN4 net: greedy policy over its own hand and all public calls."""

    def __init__(self, path: str, rule: str = "policy"):
        self.net, ck = load_fourseat_checkpoint(path, "cpu")
        self.net.eval()
        # rule "policy" or "q>M": Double iff legal and double_value > M (/100), else best other call
        self.margin = None if rule == "policy" else float(rule.removeprefix("q>"))
        if self.margin is not None and not hasattr(self.net, "double_value_head"):
            raise ValueError("q>M rules need a checkpoint with a double_value head")
        self.name = f"{Path(path).parent.name}:four" + ("" if rule == "policy" else f":{rule}")
        self.any_seat = bool(ck.get("any_seat_double", False))   # E45b: X/XX at every seat
        self.meta = {"kind": "four", "path": path, "sha256": sha256(path), "stage": ck.get("stage"),
                     "any_seat_double": self.any_seat,
                     "step": ck.get("step"), "init_sha256": ck.get("init_sha256"), "rule": rule}

    @torch.no_grad()
    def act(self, table: Table, rows: torch.Tensor) -> torch.Tensor:
        seat = table.seat_abs(rows)
        doubles = self.net.n_actions > PASS + 1            # D5OWN4X may emit Double
        competitive = self.net.stage == "D5OWN4XC"        # X/XX only where Pass would end it
        redouble = competitive and self.net.redouble
        if competitive:
            feats = competitive_features(table.history[rows, :table.t], table.dealer[rows],
                                         table.vul_ns[rows], table.vul_ew[rows], seat)
        else:
            feats = features_from_history(table.history[rows, :table.t], table.dealer[rows],
                                          table.vul_ns[rows], table.vul_ew[rows], seat, doubles)
        out = self.net(table.deals.hands[table.deal[rows], seat], feats)
        columns = [torch.arange(L)[None] > table.st.last[rows, None],
                   torch.ones(len(rows), 1, dtype=torch.bool)]
        if doubles:
            columns.append(((table.st.last[rows] >= 0) & (table.doubled[rows] == 0)
                            & (table.contract_side[rows] != seat % 2))[:, None])
        if redouble:
            columns.append(((table.st.last[rows] >= 0) & (table.doubled[rows] == 1)
                            & (table.contract_side[rows] == seat % 2))[:, None])
        legal = torch.cat(columns, 1)
        if competitive and not self.any_seat:             # feature 149: Pass would end the auction
            legal[:, DOUBLE:] &= feats[:, 149:150] > 0
        logp = policy_log_probs(out, legal)
        if self.margin is None:
            return logp.argmax(-1)
        take = legal[:, DOUBLE] & (out["double_value"] > self.margin)
        return torch.where(take, torch.full((len(rows),), DOUBLE), logp[:, :DOUBLE].argmax(-1))


class PassPlayer:
    """Always Passes. Against it, the other player bids a silent-opponent auction."""

    name = "pass"
    meta = {"kind": "pass"}

    def act(self, table: Table, rows: torch.Tensor) -> torch.Tensor:
        return torch.full((len(rows),), PASS, dtype=torch.long)


def make_player(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "pass":
        return PassPlayer()
    if kind == "four":
        path, _, rule = rest.partition(":")
        return FourSeatPlayer(path, rule or "policy")
    if kind == "zero":
        path, _, rule = rest.partition(":")
        return ZeroPlayer(path, rule or "policy")
    raise ValueError(f"unknown player spec {spec!r}")


# ------------------------------------------------------------------- match

@torch.no_grad()
def play(players, deals, deal, dealer, vul_ns, vul_ew, controller):
    table = Table(deals, deal, dealer, vul_ns, vul_ew, controller)
    while bool(table.st.alive.any()):
        if table.t >= MAX_TABLE_CALLS:
            raise AssertionError("auction did not terminate")
        alive = table.st.alive.nonzero().squeeze(1)
        side = table.seat_abs(alive) % 2
        who = table.controller[alive, side]
        action = torch.full((len(deal),), PASS, dtype=torch.long)
        for p, player in enumerate(players):
            rows = alive[who == p]
            if len(rows):
                action[rows] = player.act(table, rows)
        table.apply(action)
    return table


def boards(n_deals: int):
    deal = torch.arange(n_deals).repeat_interleave(16)
    dealer = torch.arange(4).repeat_interleave(4).repeat(n_deals)
    vul = torch.arange(4).repeat(4 * n_deals)
    return deal, dealer, vul


def run_boards(players, deals, deal, dealer, vul, chunk, on_chunk=None):
    """Both tables for every board. Returns per-board arrays and the two Tables' histories.

    Each chunk of boards is played to the end before the next starts. ``on_chunk(res, hist1,
    hist2)`` gets the finished boards so far, so a stopped robot run keeps its results."""
    out = {k: [] for k in ("ns1", "ns2", "c1", "c2", "d1", "d2", "calls1", "calls2")}
    hist1, hist2 = [], []
    for i in range(0, len(deal), chunk):
        s = slice(i, i + chunk)
        n = len(deal[s])
        vns = (vul[s] == 1) | (vul[s] == 3)
        vew = (vul[s] == 2) | (vul[s] == 3)
        # table 1: A NS (0), B EW (1); table 2 swapped. One batch of 2n rows.
        ctrl = torch.cat([torch.tensor([[0, 1]]).expand(n, 2), torch.tensor([[1, 0]]).expand(n, 2)])
        tab = play(players, deals, deal[s].repeat(2), dealer[s].repeat(2),
                   vns.repeat(2), vew.repeat(2), ctrl)
        ns, c, d = tab.ns_scores()
        calls = (tab.history >= 0).sum(1).numpy()
        for key, arr in (("ns", ns), ("c", c), ("d", d), ("calls", calls)):
            out[key + "1"].append(arr[:n])
            out[key + "2"].append(arr[n:])
        hist1.append(tab.history[:n].numpy())
        hist2.append(tab.history[n:].numpy())
        if on_chunk is not None and i + chunk < len(deal):
            on_chunk({k: np.concatenate(v) for k, v in out.items()},
                     np.concatenate(hist1), np.concatenate(hist2))
    res = {k: np.concatenate(v) for k, v in out.items()}
    return res, np.concatenate(hist1), np.concatenate(hist2)


def replay_check(deals, deal, dealer, vul, hist, ns, n_check, seed):
    """Replay sampled auctions through bridgezero's reference AuctionState."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(deal), min(n_check, len(deal)), replace=False)
    tricks = deals.tricks.numpy()
    for i in idx:
        v = int(vul[i])
        state = AuctionState.from_calls([c for c in hist[i] if c >= 0], dealer=int(dealer[i]),
                                        vul_ns=v in (1, 3), vul_ew=v in (2, 3))
        if not state.ended:
            raise AssertionError(f"replayed auction {i} did not end")
        if terminal_ns_score(state, tricks[int(deal[i])]) != int(ns[i]):
            raise AssertionError(f"score mismatch on auction {i}: {state.format_history()}")
    return int(len(idx))


def trim_history(hist: np.ndarray) -> np.ndarray:
    """Call histories as int8, cut to the longest auction (-1 = no call)."""
    width = max(1, int((hist >= 0).sum(1).max()))
    return hist[:, :width].astype(np.int8)


def summarize(res, deal, vul, deal_local, seed):
    points = res["ns1"] - res["ns2"]
    imps = imps_array(points).astype(np.float64)
    report = {
        "boards": int(len(points)),
        "imps_per_board": paired_bootstrap(imps, deal_local, seed=seed),
        "points_per_board": paired_bootstrap(points.astype(np.float64), deal_local, seed=seed),
        "boards_won_lost_tied": [int((imps > 0).sum()), int((imps < 0).sum()), int((imps == 0).sum())],
        "same_contract_both_tables": float(((res["c1"] == res["c2"]) & (res["d1"] == res["d2"])).mean()),
        "imps_by_vul": {VUL_NAMES[v]: float(imps[vul == v].mean()) for v in range(4)
                        if (vul == v).any()},
    }
    for tag, table in (("table1_A_NS", "1"), ("table2_A_EW", "2")):
        c, d = res["c" + table], res["d" + table]
        level = np.where(c >= 0, c // 5 + 1, 0)
        a_side = 0 if table == "1" else 1
        report[tag] = {
            "passed_out": float((c < 0).mean()),
            "A_declares": float(((d >= 0) & (d % 2 == a_side)).mean()),
            "B_declares": float(((d >= 0) & (d % 2 != a_side)).mean()),
            "mean_calls": float(res["calls" + table].mean()),
            "level_share": {str(lv): float((level == lv).mean()) for lv in range(8)},
        }
    return report, imps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=DATA)
    p.add_argument("--deals", type=int, default=10000, help="last N deals of the dataset")
    p.add_argument("--chunk", type=int, default=0,
                   help="boards played to the end at once (default 16000, 25 with the robot)")
    p.add_argument("--replay-check", type=int, default=300)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boards", type=int, default=0,
                   help="play a seeded random subset of N boards (0 = all)")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a, b = (make_player(s) for s in (args.a, args.b))
    remote = False
    deals = load_range(args.data, -args.deals, args.deals)
    deal, dealer, vul = boards(deals.n)
    if args.boards > 0:
        pick = torch.as_tensor(np.sort(np.random.default_rng(args.seed).choice(
            len(deal), min(args.boards, len(deal)), replace=False)))
        deal, dealer, vul = deal[pick], dealer[pick], vul[pick]

    # Symmetry check: each player against itself scores exactly zero.
    for player in (a, b):
        if getattr(player, "remote", False):
            continue
        n = min(160, len(deal))
        self_res, _, _ = run_boards([player, player], deals, deal[:n], dealer[:n], vul[:n], n)
        if np.any(self_res["ns1"] != self_res["ns2"]):
            raise AssertionError(f"{player.name} against itself is not zero")
    for player in (a, b):
        if hasattr(player, "reset_stats"):
            player.reset_stats()

    chunk = args.chunk or (25 if remote else 16000)
    planned = len(deal)

    def write(res, hist1, hist2):
        """Score and save the boards finished so far; returns the report."""
        n = len(res["ns1"])
        d, dl, vu = deal[:n].numpy(), dealer[:n].numpy(), vul[:n].numpy()
        checked = replay_check(deals, d, dl, vu, hist1, res["ns1"], args.replay_check, args.seed)
        checked += replay_check(deals, d, dl, vu, hist2, res["ns2"], args.replay_check, args.seed + 1)
        deal_local = np.unique(d, return_inverse=True)[1]   # dense ids for a board subset
        report, imps = summarize(res, d, vu, deal_local, args.seed)
        report = {"A": a.meta, "B": b.meta, "eval_first_index": deals.first_index,
                  "eval_deals": deals.n, "replayed_auctions": checked,
                  "seconds": round(time.time() - start, 1), "complete": n == planned,
                  "boards_planned": planned, **report}
        (out / "results.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(out / "boards.npz", deal_index=d + deals.first_index,
                            dealer=dl, vul=vu, imps_A=imps.astype(np.int8),
                            hist1=trim_history(hist1), hist2=trim_history(hist2),
                            **{k: v for k, v in res.items()})
        return report, imps, checked

    def progress(res, hist1, hist2):
        report, _, _ = write(res, hist1, hist2)
        r = report["imps_per_board"]
        print(f"saved {report['boards']}/{planned} boards: A IMPs/board {r['mean']:+.2f} "
              f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]", flush=True)

    res, hist1, hist2 = run_boards([a, b], deals, deal, dealer, vul, chunk,
                                   on_chunk=progress if remote else None)
    for player in (a, b):
        if hasattr(player, "close"):
            player.close()
    report, imps, checked = write(res, hist1, hist2)
    vul_np = vul.numpy()

    with (out / "sample_auctions.txt").open("w") as fh:
        rng = np.random.default_rng(args.seed)
        for i in rng.choice(len(deal), min(25, len(deal)), replace=False):
            v = int(vul_np[i])
            fh.write(f"board {i} deal {int(deal[i]) + deals.first_index} vul {VUL_NAMES[v]} "
                     f"A imps {int(imps[i]):+d}\n")
            for tag, hist, ns in (("T1 A=NS", hist1, res["ns1"]), ("T2 A=EW", hist2, res["ns2"])):
                state = AuctionState.from_calls([c for c in hist[i] if c >= 0], dealer=int(dealer[i]),
                                                vul_ns=v in (1, 3), vul_ew=v in (2, 3))
                fh.write(f"  {tag}: {state.format_history()}  => {state.final_contract()} "
                         f"NS {int(ns[i]):+d}\n")

    r = report["imps_per_board"]
    print(f"A={a.name}  B={b.name}  boards={report['boards']}")
    print(f"A IMPs/board {r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
          f"points/board {report['points_per_board']['mean']:+.1f}")
    print(f"won/lost/tied {report['boards_won_lost_tied']}  same contract "
          f"{100 * report['same_contract_both_tables']:.1f}%  by vul {report['imps_by_vul']}")
    for tag in ("table1_A_NS", "table2_A_EW"):
        t = report[tag]
        print(f"{tag}: passout {t['passed_out']:.3f} A declares {t['A_declares']:.3f} "
              f"B declares {t['B_declares']:.3f} calls {t['mean_calls']:.1f}")
    print(f"replayed {checked} auctions OK, {report['seconds']}s")


if __name__ == "__main__":
    main()
