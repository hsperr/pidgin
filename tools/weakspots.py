"""Weak-spot numbers for a tools/match.py ``boards.npz`` (A = the model under test).

- contested split: A's IMPs/board when both, one or no table had both sides bid or double
- per side (A, B): doubled-and-down tables, penalty doubles by level, punish rate
  (share of the opponents' failing contracts this side doubled), and the value of
  the side's final outbid by its combined HCP (IMPs vs letting the opponents play
  their last bid undoubled; DD tricks, rest of the auction frozen)
- board classes: B1 (A declares at both tables, doubled and down at one or more)
  and C (B declares at both tables), as IMPs per board of the match

    python tools/weakspots.py results/a_vs_b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridgezero.bridge.deals import load_dataset  # noqa: E402
from bridgezero.bridge.scoring import contract_score  # noqa: E402

DATA = "data/dds_results_100M.npy"
PASS, DOUBLE, REDOUBLE = 35, 36, 37
BID2TAB = np.array([3, 2, 1, 0, 4])              # bid strain C,D,H,S,NT -> tricks S,H,D,C,NT
IMP_LOWER = [20, 50, 90, 130, 170, 220, 270, 320, 370, 430, 500, 600, 750, 900, 1100, 1300,
             1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000]
HCP_BANDS = [(0, 15, "<=15"), (16, 17, "16-17"), (18, 19, "18-19"), (20, 21, "20-21"),
             (22, 40, "22+")]
# SCORE[vul, doubled 0/1/2, contract, tricks] = declarer score
SCORE = np.array([[[[contract_score(c // 5 + 1, c % 5, k, x, bool(v)) for k in range(14)]
                    for c in range(35)] for x in range(3)] for v in range(2)])


def imps(points):
    points = np.asarray(points)
    return np.sign(points) * np.searchsorted(IMP_LOWER, np.abs(points), side="right")


def decode(hist: np.ndarray, dealer: np.ndarray) -> dict:
    """Per-table auction facts. Sides: 0 = NS, 1 = EW; -1 = none."""
    n = len(hist)
    out = {k: np.full(n, -1) for k in ("doubled", "prev_c", "prev_decl", "outbid_side")}
    out["bid_sides"] = np.zeros((n, 2), bool)
    xs = []                                          # (table, doubling side, doubled level)
    for i in range(n):
        d = int(dealer[i])
        first = {}                                   # (side, strain) -> first seat to name it
        last_bid, last_side, doubled = -1, -1, 0
        prev = (-1, -1, -1)                          # opponents' last bid: contract, declarer, side
        for j, c in enumerate(hist[i]):
            if c < 0:
                break
            seat = (d + j) % 4
            side = seat % 2
            if c < PASS:
                out["bid_sides"][i, side] = True
                first.setdefault((side, c % 5), seat)
                if last_bid >= 0 and last_side != side:
                    prev = (last_bid, first[(last_side, last_bid % 5)], last_side)
                last_bid, last_side, doubled = int(c), side, 0
            elif c == DOUBLE:
                out["bid_sides"][i, side] = True
                doubled = 1
                xs.append((i, side, last_bid // 5 + 1))
            elif c == REDOUBLE:
                doubled = 2
        out["doubled"][i] = doubled
        if last_bid >= 0 and prev[2] >= 0:
            out["prev_c"][i], out["prev_decl"][i] = prev[0], prev[1]
            out["outbid_side"][i] = last_side
    out["x"] = np.array(xs, dtype=np.int64).reshape(-1, 3)
    return out


def table_score(tricks, vul_ns, vul_ew, c, decl, doubled):
    """NS score of contract ``c`` by ``decl`` (0 where c < 0)."""
    ok = c >= 0
    cc, dd = np.where(ok, c, 0), np.where(ok, decl, 0)
    t = tricks[np.arange(len(c)), dd, BID2TAB[cc % 5]]
    vul = np.where(dd % 2 == 0, vul_ns, vul_ew).astype(int)
    raw = SCORE[vul, doubled, cc, t]
    return np.where(ok, np.where(dd % 2 == 0, raw, -raw), 0), np.where(ok, t - (cc // 5 + 7), 0)


def analyse(z: dict, owners: np.ndarray, tricks: np.ndarray) -> dict:
    n = len(z["imps_A"])
    board_imps = z["imps_A"].astype(float)
    dealer, vul = z["dealer"], z["vul"]
    vul_ns, vul_ew = np.isin(vul, (1, 3)), np.isin(vul, (2, 3))
    tr = tricks.astype(np.int64)
    pts = np.where(np.arange(52) % 13 < 4, 4 - np.arange(52) % 13, 0)
    hcp = np.stack([((owners == s) * pts).sum(1) for s in range(4)], 1)
    side_hcp = np.stack([hcp[:, 0] + hcp[:, 2], hcp[:, 1] + hcp[:, 3]], 1)

    tables = []
    for t, a_side in ((1, 0), (2, 1)):
        f = decode(z[f"hist{t}"].astype(int), dealer)
        c, decl = z[f"c{t}"].astype(int), z[f"d{t}"].astype(int)
        ns, res = table_score(tr, vul_ns, vul_ew, c, decl, np.maximum(f["doubled"], 0))
        if not np.allclose(ns, z[f"ns{t}"]):
            raise AssertionError(f"table {t}: rescoring does not reproduce ns{t}")
        prev_ns, _ = table_score(tr, vul_ns, vul_ew, f["prev_c"], f["prev_decl"],
                                 np.zeros(n, int))
        tables.append(dict(f, c=c, decl_side=np.where(c >= 0, decl % 2, -1), res=res, ns=ns,
                           prev_ns=prev_ns, a_side=a_side))

    report = {"boards": int(n), "imps_per_board": float(board_imps.mean())}
    contested = sum(tb["bid_sides"].all(1).astype(int) for tb in tables)
    report["contested"] = {name: {"share": float((contested == k).mean()),
                                  "imps": float(board_imps[contested == k].mean())
                                  if (contested == k).any() else None}
                           for k, name in ((2, "both"), (1, "one"), (0, "none"))}

    for who in ("A", "B"):
        dd_tables = x_by_level = punish_hit = punish_all = 0
        x_levels = np.zeros(8)
        dd_imps = 0.0
        outbid = {lab: [] for _, _, lab in HCP_BANDS}
        for tb in tables:
            side = tb["a_side"] if who == "A" else 1 - tb["a_side"]
            sign = 1 if side == 0 else -1
            mine = tb["decl_side"] == side
            theirs = tb["decl_side"] == 1 - side
            dd = mine & (tb["doubled"] > 0) & (tb["res"] < 0)
            dd_tables += int(dd.sum())
            dd_imps += float(board_imps[dd].sum()) * (1 if who == "A" else -1)
            x = tb["x"][tb["x"][:, 1] == side]
            x_levels += np.bincount(np.minimum(x[:, 2], 7), minlength=8)
            down = theirs & (tb["res"] < 0)
            punish_all += int(down.sum())
            punish_hit += int((down & (tb["doubled"] > 0)).sum())
            ob = tb["outbid_side"] == side
            value = imps(sign * (tb["ns"] - tb["prev_ns"]))
            for lo, hi, lab in HCP_BANDS:
                m = ob & (side_hcp[:, side] >= lo) & (side_hcp[:, side] <= hi)
                outbid[lab].append((value[m], tb["doubled"][m] > 0, tb["res"][m] < 0))
        per_1k = 1000 / (2 * n)
        report[who] = {
            "doubled_down_per_1k": dd_tables * per_1k,
            "doubled_down_imps_per_board": dd_imps / n,
            "x_per_1k_by_level": {str(lv): float(x_levels[lv] * per_1k) for lv in range(1, 6)},
            "punish_rate": punish_hit / punish_all if punish_all else None,
            "outbid": {},
        }
        for lab, parts in outbid.items():
            v, dbl, dn = (np.concatenate(p) for p in zip(*parts))
            report[who]["outbid"][lab] = ({"n": int(len(v)), "imps": float(v.mean()),
                                           "doubled": float(dbl.mean()), "down": float(dn.mean())}
                                          if len(v) else {"n": 0})

    a_both = (tables[0]["decl_side"] == 0) & (tables[1]["decl_side"] == 1)
    b_both = (tables[0]["decl_side"] == 1) & (tables[1]["decl_side"] == 0)
    a_dd = np.zeros(n, bool)
    for tb in tables:
        a_dd |= (tb["decl_side"] == tb["a_side"]) & (tb["doubled"] > 0) & (tb["res"] < 0)
    for name, m in (("B1", a_both & a_dd), ("C", b_both)):
        report[name] = {"share": float(m.mean()),
                        "imps_per_board": float(board_imps[m].sum() / n),
                        "mean": float(board_imps[m].mean()) if m.any() else None}
    return report


def openings(z: dict, owners: np.ndarray, who: str = "A") -> dict:
    """``who``'s opening calls from the match auctions (both tables).

    ``calls``: per opening bid, count, share of all openings, HCP (mean, 10th and 90th
    percentile) and mean suit lengths S/H/D/C. ``open_rate``: share of hands opened by
    HCP band when ``who`` is dealer (the first to speak), which every table has once.
    """
    pts = np.where(np.arange(52) % 13 < 4, 4 - np.arange(52) % 13, 0)
    suit = np.arange(52) // 13
    rows, dealer_rows = [], []
    for key, a_side in (("hist1", 0), ("hist2", 1)):
        side = a_side if who == "A" else 1 - a_side
        for i, hist in enumerate(z[key]):
            dealer = int(z["dealer"][i])
            for j, call in enumerate(int(x) for x in hist if x >= 0):
                if call == PASS:
                    continue
                seat = (dealer + j) % 4
                if seat % 2 == side:
                    hand = owners[i] == seat
                    rows.append((call, int(pts[hand].sum()),
                                 *[int((hand & (suit == k)).sum()) for k in range(4)]))
                break
            if dealer % 2 == side:
                hand = owners[i] == dealer
                first = next((int(x) for x in hist if x >= 0), PASS)
                dealer_rows.append((int(pts[hand].sum()), first != PASS))
    names = [f"{c // 5 + 1}{'CDHS'[c % 5] if c % 5 < 4 else 'NT'}" for c in range(35)]
    calls = {}
    if rows:
        r = np.array(rows)
        for c in np.unique(r[:, 0]):
            m = r[r[:, 0] == c]
            calls[names[c]] = {"n": int(len(m)), "share": float(len(m) / len(r)),
                               "hcp": float(m[:, 1].mean()),
                               "hcp_p10": float(np.percentile(m[:, 1], 10)),
                               "hcp_p90": float(np.percentile(m[:, 1], 90)),
                               "len": [float(v) for v in m[:, 2:].mean(0)]}
    d = np.array(dealer_rows).reshape(-1, 2)
    open_rate = {}
    for lo, hi, lab in ((0, 5, "0-5"), (6, 9, "6-9"), (10, 11, "10-11"), (12, 14, "12-14"),
                        (15, 17, "15-17"), (18, 40, "18+")):
        m = (d[:, 0] >= lo) & (d[:, 0] <= hi)
        open_rate[lab] = {"n": int(m.sum()), "rate": float(d[m, 1].mean()) if m.any() else None}
    return {"openings": int(len(rows)), "calls": calls, "open_rate": open_rate}


def run(match_dir: Path, data: str = DATA) -> dict:
    z = dict(np.load(match_dir / "boards.npz"))
    owners, tricks = load_dataset(str(ROOT / data) if not Path(data).is_absolute() else data)
    first, last = int(z["deal_index"].min()), int(z["deal_index"].max()) + 1
    ow = np.asarray(owners[first:last])[z["deal_index"] - first]
    tr = np.asarray(tricks[first:last])[z["deal_index"] - first]
    report = analyse(z, ow, tr)
    (match_dir / "weakspots.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("match_dir", type=Path)
    p.add_argument("--data", default=DATA)
    args = p.parse_args()
    r = run(args.match_dir, args.data)
    print(f"boards {r['boards']}  A IMPs/board {r['imps_per_board']:+.3f}")
    print("contested " + "  ".join(f"{k} {v['share']:.2f} ({v['imps']:+.2f})"
                                   for k, v in r["contested"].items() if v["imps"] is not None))
    for who in ("A", "B"):
        s = r[who]
        print(f"{who}: doubled&down/1k {s['doubled_down_per_1k']:.1f} "
              f"({s['doubled_down_imps_per_board']:+.3f}/board)  punish rate "
              f"{s['punish_rate'] or 0:.2f}  X/1k by level "
              + " ".join(f"{k}:{v:.1f}" for k, v in s["x_per_1k_by_level"].items()))
        print("   outbid by HCP " + "  ".join(
            f"{k} {v['imps']:+.2f} (X {v['doubled']:.2f})" for k, v in s["outbid"].items() if v["n"]))
    print(f"B1 {r['B1']['imps_per_board']:+.3f}/board ({r['B1']['share']:.3f})  "
          f"C {r['C']['imps_per_board']:+.3f}/board ({r['C']['share']:.3f})")


if __name__ == "__main__":
    main()
