"""Local training dashboard: metrics, openings, and IMP matches against public
checkpoints and built-in rule bidders. Matches include weak-spot measurements.

    python tools/dashboard.py --runs runs/training
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simplicity  # noqa: E402
import weakspots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("dashboard.html")
BELIEF_PAGE = Path(__file__).with_name("belief.html")
TOURNEY_PAGE = Path(__file__).with_name("tourney.html")
TOURNEY_FILE = Path.home() / "code/bridge_new/experiments/belief_multi/runs/tourney/standings.json"
# --belief-runs: belief-net runs (bridge_new E57), one folder per run with log.jsonl + args.json.
BELIEF_RUNS = Path.home() / "code/bridge_new/experiments/belief_multi/runs"
OPENINGS_FILE = "openings_latest.txt"
jobs: dict[str, dict] = {}                       # run dir -> {"state", "started", "ckpt"}
lock = threading.Lock()

RULE_BOTS = ("sayc", "weakclub", "happy")
D_REF: Path | None = None
# --opponents FILE: extra opponents for this machine only, e.g. checkpoints or bots outside
# this repo. label -> {"spec", "boards", optional "match_py" and "cwd" for another match script}.
EXTRA: dict[str, dict] = {}
FOUR_BOARDS = 20000
DATA = ROOT / "data" / "dds_results_100M.npy"
ANALYSIS_DEALS = 10000
MATCH_THREADS = int(os.environ.get("DASHBOARD_MATCH_THREADS", "2"))
IMP_FILE = "imps_history.jsonl"
SIMPLE_FILE = "simplicity.jsonl"
imp_jobs: dict[str, dict] = {}                   # run dir -> {"state", "ckpt", "done", "total", "current"}


def flatten(record: dict, prefix: str = "") -> dict:
    """Numbers only, nested keys joined with '.'; lists and strings are dropped."""
    out = {}
    for key, value in record.items():
        name = prefix + key
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = value
    return out


def list_runs(base: Path) -> list[dict]:
    base = base.resolve()
    runs = [p.parent for p in base.rglob("train_log.jsonl") if "backup" not in str(p)]
    runs.sort(key=lambda p: (p / "train_log.jsonl").stat().st_mtime, reverse=True)
    return [{"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
             "mtime": (p / "train_log.jsonl").stat().st_mtime,
             "description": run_description(p)} for p in runs]


def run_description(run: Path) -> str:
    """``description.txt`` in the run folder, else ``--description`` from the newest run json."""
    text = run / "description.txt"
    if text.exists():
        return text.read_text().strip()
    for meta in sorted(run.glob("run*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            if desc := json.loads(meta.read_text()).get("description"):
                return desc
        except (OSError, ValueError):
            continue
    return ""


def read_log(run: Path) -> list[dict]:
    """Flattened log rows. Simplicity numbers are keyed ``simplicity.*``: the trainer logs
    them under ``validation.fourseat.simplicity``; runs from before that have them in
    ``simplicity.jsonl`` (one row per checkpoint, from tools/simplicity.py)."""
    rows = []
    for line in (run / "train_log.jsonl").read_text().splitlines():
        try:
            row = flatten(json.loads(line))
        except json.JSONDecodeError:                 # a line being written right now
            continue
        rows.append({k.replace("validation.fourseat.simplicity.", "simplicity."): v
                     for k, v in row.items()})
    by_step = {r.get("step"): r for r in rows}
    for extra in read_jsonl(run / SIMPLE_FILE):
        if "error" in extra or not isinstance(extra.get("step"), int):
            continue
        if extra["step"] not in by_step:             # a checkpoint between two eval rows
            by_step[extra["step"]] = {"step": extra["step"]}
            rows.append(by_step[extra["step"]])
        target = by_step[extra["step"]]
        for key, value in extra.items():
            if key not in ("step", "auctions") and isinstance(value, (int, float)):
                target.setdefault(f"simplicity.{key}", value)
    rows.sort(key=lambda r: r.get("step", -1))
    return rows


def newest_checkpoint(run: Path) -> Path | None:
    steps = [(int(m.group(1)), p) for p in run.glob("ckpt_step*.pt")
             if (m := re.fullmatch(r"ckpt_step(\d+)\.pt", p.name))]
    if steps:
        return max(steps)[1]
    for name in ("last.pt", "best.pt"):
        if (run / name).exists():
            return run / name
    return None


def checkpoint_step(ckpt: Path) -> int | None:
    if m := re.fullmatch(r"ckpt_step(\d+)\.pt", ckpt.name):
        return int(m.group(1))
    try:
        import torch
        return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("step"))
    except Exception:                                # noqa: BLE001 - unknown file layout
        return None


def run_simplicity(run: Path, ckpt: Path) -> None:
    """Self-play simplicity numbers (training/simplicity.py) for ``ckpt``, appended to
    ``simplicity.jsonl`` so the charts pick up numbers added after the run was logged."""
    step = checkpoint_step(ckpt)
    if step is None:
        return
    numbers = simplicity.self_play(ckpt, ANALYSIS_DEALS, MATCH_THREADS, str(DATA))
    with (run / SIMPLE_FILE).open("a") as f:
        f.write(json.dumps({"step": step, "ckpt": ckpt.name, **numbers}) + "\n")


def run_openings(run: Path, ckpt: Path) -> None:
    """Opening table plus the simplicity numbers for ``ckpt``."""
    cmd = [sys.executable, str(ROOT / "tools" / "openings.py"), str(ckpt),
           "--examples", "3", "--threads", "2", "--data", str(DATA),
           "--deals", str(ANALYSIS_DEALS)]
    state = "failed"
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        text = proc.stdout if proc.returncode == 0 else proc.stdout + "\n" + proc.stderr
        stamp = time.strftime("%Y-%m-%d %H:%M")
        (run / OPENINGS_FILE).write_text(f"# {ckpt.name}  (run at {stamp})\n{text}")
        state = "done" if proc.returncode == 0 else "failed"
        run_simplicity(run, ckpt)
    except (OSError, subprocess.CalledProcessError) as exc:
        with lock:
            jobs[str(run)]["error"] = str(exc)
    finally:
        with lock:
            jobs[str(run)].update(state=state, ckpt=ckpt.name)


def checkpoint_steps(run: Path) -> list[tuple[int, Path]]:
    return sorted((int(m.group(1)), p) for p in run.glob("ckpt_step*.pt")
                  if (m := re.fullmatch(r"ckpt_step(\d+)\.pt", p.name)))


def imp_opponents(run: Path, newest_step: int) -> list[tuple[str, str, int]]:
    """(label, match.py player spec, boards) for every opponent."""
    out = []
    seen = set()
    steps = checkpoint_steps(run)
    for frac in (0.25, 0.5):
        target = newest_step * frac
        if steps:
            step, path = min(steps, key=lambda sp: abs(sp[0] - target))
            if step < newest_step and path not in seen:
                seen.add(path)
                out.append((f"this run step {step} (~{frac:.0%})", f"four:{path}", FOUR_BOARDS))
    if D_REF is not None and D_REF.exists():
        # D itself, and D with a DD-oracle Double: how much do thin bids cost against a punisher?
        out += [("reference model", f"four:{D_REF}", FOUR_BOARDS),
                ("reference + DD punisher", f"punish:{D_REF}", FOUR_BOARDS)]
    out += [(f"rule bot {style}", f"rule:{style}", FOUR_BOARDS) for style in RULE_BOTS]
    out += [(label, o["spec"], int(o.get("boards", FOUR_BOARDS))) for label, o in EXTRA.items()]
    return out


def load_extra_opponents(file: Path) -> dict[str, dict]:
    """``--opponents`` JSON: a list of {label, spec, boards?, match_py?, cwd?}; paths may be
    relative to the file."""
    base = file.resolve().parent
    out = {}
    for o in json.loads(file.read_text()):
        entry = {"spec": o["spec"], "boards": int(o.get("boards", FOUR_BOARDS))}
        for key in ("match_py", "cwd"):
            if o.get(key):
                entry[key] = (base / o[key]).resolve()
        out[o["label"]] = entry
    return out


def run_imps(run: Path, step: int, ckpt: Path) -> None:
    opponents = imp_opponents(run, step)
    with lock:
        imp_jobs[str(run)].update(total=len(opponents))
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for i, (label, spec, boards) in enumerate(opponents):
        with lock:
            imp_jobs[str(run)].update(done=i, current=label)
        out = run / "matches_dashboard" / f"step{step}_vs_{re.sub(r'[^A-Za-z0-9]+', '_', label)}"
        extra = EXTRA.get(label, {})
        cmd = [sys.executable, "-u", str(extra.get("match_py", ROOT / "tools" / "match.py")),
               "--a", f"four:{ckpt}", "--b", spec,
               "--boards", str(boards), "--threads", str(MATCH_THREADS), "--out", str(out),
               "--data", str(DATA), "--deals", str(ANALYSIS_DEALS)]
        row = {"when": stamp, "step": step, "ckpt": ckpt.name, "opponent": label,
               "boards": boards}
        try:
            proc = subprocess.run(cmd, cwd=extra.get("cwd", ROOT), capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError((proc.stderr or proc.stdout)[-400:])
            report = json.loads((out / "results.json").read_text())
            res = report["imps_per_board"]
            row.update(imps=res["mean"], ci95=res["ci95"], boards=report["boards"])
            row["weak"] = weak_summary(weakspots.run(out, str(DATA)))
        except (OSError, KeyError, json.JSONDecodeError, RuntimeError, AssertionError) as exc:
            row["error"] = str(exc)[-400:]
        with (run / IMP_FILE).open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    with lock:
        imp_jobs[str(run)].update(state="done", done=len(opponents), current=None)


def weak_summary(report: dict) -> dict:
    """The few weak-spot numbers the dashboard shows (tools/weakspots.py has the rest)."""
    a, b = report["A"], report["B"]
    light = [a["outbid"][k] for k in ("<=15", "16-17") if a["outbid"][k]["n"]]
    n = sum(o["n"] for o in light)
    return {"dd_A": a["doubled_down_per_1k"], "dd_B": b["doubled_down_per_1k"],
            "punish_A": a["punish_rate"], "punish_B": b["punish_rate"],
            "x_low_A": sum(a["x_per_1k_by_level"][k] for k in ("1", "2")),
            "outbid_light_A": sum(o["imps"] * o["n"] for o in light) / n if n else None,
            "B1": report["B1"]["imps_per_board"], "C": report["C"]["imps_per_board"],
            "contested": report["contested"]["both"]["imps"],
            "uncontested": report["contested"]["none"]["imps"]}


def read_jsonl(file: Path) -> list[dict]:
    if not file.exists():
        return []
    rows = []
    for line in file.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a result is being appended now
                continue
    return rows


def list_belief_runs() -> list[dict]:
    if not BELIEF_RUNS.is_dir():
        return []
    runs = sorted((p for p in BELIEF_RUNS.iterdir() if (p / "log.jsonl").exists()),
                  key=lambda p: (p / "log.jsonl").stat().st_mtime, reverse=True)
    out = []
    for p in runs:
        try:
            args = json.loads((p / "args.json").read_text())
        except (OSError, ValueError):
            args = {}
        out.append({"name": p.name, "args": args})
    return out


class Handler(BaseHTTPRequestHandler):
    base: Path = ROOT / "runs"

    def send(self, body: bytes, kind: str = "application/json", code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def run_dir(self, query: dict) -> Path | None:
        run = (ROOT / query.get("run", [""])[0]).resolve()
        if not run.is_relative_to(self.base.resolve()) or not (run / "train_log.jsonl").exists():
            self.send(b'{"error": "unknown run"}', code=404)
            return None
        return run

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/tourney":
            self.send(TOURNEY_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/tourney":
            self.send(TOURNEY_FILE.read_bytes() if TOURNEY_FILE.exists() else b'{"error": "no standings yet"}')
        elif url.path == "/belief":
            self.send(BELIEF_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/belief/runs":
            self.send(json.dumps(list_belief_runs()).encode())
        elif url.path == "/api/belief/log":
            name = query.get("run", [""])[0]
            run = BELIEF_RUNS / name
            if not name or "/" in name or ".." in name or not (run / "log.jsonl").exists():
                return self.send(b'{"error": "unknown run"}', code=404)
            self.send(json.dumps(read_jsonl(run / "log.jsonl")).encode())
        elif url.path == "/api/belief/analysis":
            name = query.get("run", [""])[0]
            f = BELIEF_RUNS / name / "analysis.json"
            if not name or "/" in name or ".." in name or not f.exists():
                return self.send(b"{}")
            self.send(f.read_bytes())
        elif url.path == "/api/runs":
            self.send(json.dumps(list_runs(self.base)).encode())
        elif url.path == "/api/log":
            if run := self.run_dir(query):
                self.send(json.dumps(read_log(run)).encode())
        elif url.path == "/api/openings":
            if run := self.run_dir(query):
                file = run / OPENINGS_FILE
                with lock:
                    job = jobs.get(str(run), {})
                ckpt = newest_checkpoint(run)
                self.send(json.dumps({
                    "state": job.get("state", "idle"), "running_ckpt": job.get("ckpt"),
                    "newest_ckpt": ckpt.name if ckpt else None, "error": job.get("error"),
                    "text": file.read_text() if file.exists() else ""}).encode())
        elif url.path == "/api/imps":
            if run := self.run_dir(query):
                rows = read_jsonl(run / IMP_FILE)
                with lock:
                    job = dict(imp_jobs.get(str(run), {"state": "idle"}))
                self.send(json.dumps({"job": job, "rows": rows}).encode())
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/imps":
            return self.start_imps(parse_qs(url.query))
        if url.path != "/api/openings":
            return self.send(b"not found", "text/plain", 404)
        if not (run := self.run_dir(parse_qs(url.query))):
            return
        if (run / "result.json").exists():
            return self.send(b'{"error": "opening analysis starts with stage 2 four-seat checkpoints"}', code=400)
        ckpt = newest_checkpoint(run)
        if ckpt is None:
            return self.send(b'{"error": "no checkpoint in this run"}', code=400)
        with lock:
            if jobs.get(str(run), {}).get("state") == "running":
                return self.send(b'{"state": "running"}')
            jobs[str(run)] = {"state": "running", "ckpt": ckpt.name}
        threading.Thread(target=run_openings, args=(run, ckpt), daemon=True).start()
        self.send(json.dumps({"state": "running", "ckpt": ckpt.name}).encode())

    def start_imps(self, query: dict) -> None:
        if not (run := self.run_dir(query)):
            return
        steps = checkpoint_steps(run)
        if not steps:
            return self.send(b'{"error": "no ckpt_step*.pt in this run"}', code=400)
        step, ckpt = steps[-1]
        if not imp_opponents(run, step):
            return self.send(b'{"error": "no match opponents available"}', code=400)
        with lock:
            if imp_jobs.get(str(run), {}).get("state") == "running":
                return self.send(b'{"state": "running"}')
            imp_jobs[str(run)] = {"state": "running", "ckpt": ckpt.name, "done": 0,
                                  "total": 0, "current": None}
        threading.Thread(target=run_imps, args=(run, step, ckpt), daemon=True).start()
        self.send(json.dumps({"state": "running", "ckpt": ckpt.name}).encode())

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    global DATA, ANALYSIS_DEALS, D_REF, EXTRA, BELIEF_RUNS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--runs", default="runs", help="folder searched for train_log.jsonl files")
    p.add_argument("--data", default=str(DATA), help="deal dataset used by analysis jobs")
    p.add_argument("--deals", type=int, default=10000, help="last N deals for analysis jobs")
    p.add_argument("--reference", type=Path, help="optional public checkpoint for reference and DD-punisher matches")
    p.add_argument("--opponents", type=Path,
                   help="JSON list of extra opponents for this machine (see load_extra_opponents)")
    p.add_argument("--belief-runs", type=Path, default=BELIEF_RUNS,
                   help="folder of belief-net runs shown on /belief")
    args = p.parse_args()
    BELIEF_RUNS = args.belief_runs.expanduser().resolve()
    if args.opponents is not None:
        EXTRA = load_extra_opponents(args.opponents)
    if args.reference is not None and not args.reference.is_file():
        p.error("--reference checkpoint does not exist")
    D_REF = args.reference.resolve() if args.reference is not None else None
    if args.deals < 1:
        p.error("--deals must be positive")
    DATA = Path(args.data).resolve()
    ANALYSIS_DEALS = args.deals
    Handler.base = (ROOT / args.runs).resolve()
    print(f"dashboard: http://localhost:{args.port}  (runs under {Handler.base})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
