"""The local dashboard offers matches without a sibling experiment checkout."""

from pathlib import Path

from tools import dashboard


def test_imp_opponents_use_public_checkpoints(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    for step in (100, 200, 400):
        (run / f"ckpt_step{step}.pt").touch()

    monkeypatch.setattr(dashboard, "D_REF", tmp_path / "missing_d.pt")
    opponents = [o for o in dashboard.imp_opponents(run, 400) if not o[1].startswith("rule:")]
    assert len(opponents) == 2
    assert all(spec.startswith("four:") for _, spec, _ in opponents)
    assert {Path(spec[5:]).name for _, spec, _ in opponents} == {
        "ckpt_step100.pt", "ckpt_step200.pt"}


def test_read_jsonl_ignores_a_partial_live_write(tmp_path):
    file = tmp_path / "imps_history.jsonl"
    file.write_text('{"step": 1}\n{"step":')
    assert dashboard.read_jsonl(file) == [{"step": 1}]


def test_list_runs_accepts_a_relative_path(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    run = tmp_path / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "train_log.jsonl").write_text("")
    monkeypatch.chdir(tmp_path)
    assert dashboard.list_runs(Path("runs"))[0]["path"] == "runs/trial"


def test_weakspots_decode_finds_outbid_and_double():
    import numpy as np
    from tools import weakspots
    # dealer N: 1H(N) 1S(E) 2H(S) X(W) P P P -> 2HX by N; E-W's last bid 1S by E
    hist = np.array([[2, 3, 7, 36, 35, 35, 35, -1]])
    f = weakspots.decode(hist, np.array([0]))
    assert f["doubled"][0] == 1
    assert f["prev_c"][0] == 3 and f["prev_decl"][0] == 1 and f["outbid_side"][0] == 0
    assert f["bid_sides"][0].all()
    assert f["x"].tolist() == [[0, 1, 2]]


def test_imp_opponents_do_not_repeat_a_checkpoint(tmp_path):
    (tmp_path / "ckpt_step100.pt").touch()
    (tmp_path / "ckpt_step400.pt").touch()
    specs = [spec for _, spec, _ in dashboard.imp_opponents(tmp_path, 400)]
    assert len(specs) == len(set(specs))


def test_grounding_log_is_available_to_dashboard(tmp_path):
    (tmp_path / "train_log.jsonl").write_text(
        '{"step": 2, "best_step": 0, "validation": {"score": 12.5}}\n')
    assert dashboard.read_log(tmp_path) == [
        {"step": 2, "best_step": 0, "validation.score": 12.5}]


def test_match_history_records_played_boards_and_selected_dataset(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    run = tmp_path / "run"
    run.mkdir()
    dataset = tmp_path / "tiny.npz"
    monkeypatch.setattr(dashboard, "DATA", dataset)
    monkeypatch.setattr(dashboard, "imp_opponents", lambda *_: [("parent", "four:parent.pt", 20000)])
    monkeypatch.setattr(dashboard, "weak_summary", lambda report: {})
    seen = []
    monkeypatch.setattr(dashboard.weakspots, "run", lambda out, data: seen.append(data) or {})

    def match(cmd, **kwargs):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "results.json").write_text(json.dumps(
            {"boards": 256, "imps_per_board": {"mean": 0, "ci95": [0, 0]}}))
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(dashboard.subprocess, "run", match)
    dashboard.imp_jobs[str(run)] = {"state": "running"}
    dashboard.run_imps(run, 2, run / "last.pt")
    assert dashboard.read_jsonl(run / dashboard.IMP_FILE)[0]["boards"] == 256
    assert seen == [str(dataset)]
    assert dashboard.imp_jobs[str(run)]["state"] == "done"


def test_opening_job_launch_failure_releases_running_state(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("could not launch Python")
    monkeypatch.setattr(dashboard.subprocess, "run", fail)
    dashboard.jobs[str(tmp_path)] = {"state": "running"}
    dashboard.run_openings(tmp_path, tmp_path / "last.pt")
    assert dashboard.jobs[str(tmp_path)]["state"] == "failed"


def test_extra_opponents_come_from_a_local_file(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    file = tmp_path / "opponents.json"
    file.write_text('[{"label": "bot X", "spec": "x:1", "boards": 300, '
                    '"match_py": "other/match.py", "cwd": "other"}]')
    extra = dashboard.load_extra_opponents(file)
    assert extra["bot X"] == {"spec": "x:1", "boards": 300, "match_py": other / "match.py",
                              "cwd": other}
    monkeypatch.setattr(dashboard, "EXTRA", extra)
    run = tmp_path / "run"
    run.mkdir()
    assert ("bot X", "x:1", 300) in dashboard.imp_opponents(run, 0)


def test_run_description_prefers_text_file_then_run_json(tmp_path):
    import json
    run = tmp_path / "r"
    run.mkdir()
    assert dashboard.run_description(run) == ""
    (run / "run.json").write_text(json.dumps({"description": "from args"}))
    assert dashboard.run_description(run) == "from args"
    (run / "description.txt").write_text("hand written\n")
    assert dashboard.run_description(run) == "hand written"


def test_simplicity_rows_fill_missing_numbers_and_add_checkpoint_steps(tmp_path):
    import json
    run = tmp_path / "r"
    run.mkdir()
    (run / "train_log.jsonl").write_text(json.dumps(
        {"step": 2000, "validation": {"fourseat": {"simplicity": {"code_words_per_100": 4.0}}}}) + "\n")
    (run / "simplicity.jsonl").write_text(
        json.dumps({"step": 2000, "code_words_per_100": 9.0, "light_open_share": 0.4}) + "\n"
        + json.dumps({"step": 1000, "light_open_share": 0.2}) + "\n")
    rows = dashboard.read_log(run)
    assert [r["step"] for r in rows] == [1000, 2000]
    assert rows[1]["simplicity.code_words_per_100"] == 4.0          # the trainer's number wins
    assert rows[1]["simplicity.light_open_share"] == 0.4
    assert rows[0]["simplicity.light_open_share"] == 0.2
