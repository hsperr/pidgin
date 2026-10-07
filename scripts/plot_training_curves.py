"""Render the documentation's training curves from the committed numeric snapshot.

    python -m pip install matplotlib
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --preview-dir /tmp/pidgin-curves

Reads docs/data/training_curves.json and writes one multi-panel SVG per model to
docs/figures/. Each model has training stages; "steps" panels (the default) are stacked
on one shared step axis with the stages laid end to end and a line at each stage change.
"xy" and "bars" panels show single measurements and share one row below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#1f6fb2", "#d9661f", "#2e8b57", "#7b4fc4", "#c0392b", "#0f8a8a"]
INK, MUTED, GRID, EDGE = "#1e293b", "#64748b", "#e2e8f0", "#cbd5e1"
STAGE_TINT = ["#f8fafc", "#eef2f7"]


def smooth(ys: list[float], window: int) -> list[float]:
    """Centred rolling mean; the window shrinks at the ends."""
    half = window // 2
    out = []
    for i in range(len(ys)):
        part = ys[max(0, i - half):i + half + 1]
        out.append(sum(part) / len(part))
    return out


def kfmt(v, _):
    return "0" if v == 0 else f"{v / 1000:g}k"


def style_axis(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(EDGE)
    ax.tick_params(labelsize=8.5, color=EDGE)


def draw_series(ax, s: dict, ylim) -> None:
    color = COLORS[s.get("color", 0) % len(COLORS)]
    if not s["points"]:
        return
    xs, ys = zip(*s["points"])
    style = s.get("style", "line")
    if style == "markers":
        ax.plot(xs, ys, linestyle="none", marker="D", markersize=5, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        return
    window = s.get("smooth", 1)
    if window > 1:
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)
        ys = smooth(list(ys), window)
    marker = "o" if style == "both" else None
    ax.plot(xs, ys, color=color, linewidth=1.8, marker=marker, markersize=3, zorder=3)
    if ylim and ylim[0] is not None:
        for x, y in s["points"]:
            if y < ylim[0]:
                ax.annotate(f"{y:.3g}", (x, ylim[0]), xytext=(4, 6), textcoords="offset points",
                            fontsize=7.5, color=color, va="bottom")
                ax.plot([x], [ylim[0]], marker="v", color=color, markersize=6, clip_on=False, zorder=5)


def legend_entries(panel: dict) -> list[tuple[str, str, str]]:
    seen, out = set(), []
    for s in panel.get("series", []):
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        out.append((s["label"], COLORS[s.get("color", 0) % len(COLORS)], s.get("style", "line")))
    return out


def legend_handles(panel: dict) -> list:
    handles = []
    for label, color, style in legend_entries(panel):
        if style == "markers":
            h = plt.Line2D([], [], linestyle="none", marker="D", markersize=5, color=color, label=label)
        else:
            h = plt.Line2D([], [], color=color, linewidth=1.8, label=label)
        handles.append(h)
    if panel.get("ref_label"):
        handles.append(plt.Line2D([], [], color=MUTED, linewidth=1, linestyle="--",
                                  label=panel["ref_label"]))
    return handles


def stage_offsets(stages: list[dict]) -> list[int]:
    out, total = [], 0
    for st in stages:
        out.append(total)
        total += st["steps"]
    return out + [total]


def joined_series(panel: dict, offsets: list[int]) -> list[dict]:
    """One series per label, its stages laid end to end on the shared step axis."""
    gap = 0.05 * offsets[-1]
    by_label: dict[str, dict] = {}
    for s in sorted(panel["series"], key=lambda s: s["stage"]):
        if not s["points"]:
            continue
        pts = [(x + offsets[s["stage"]], y) for x, y in s["points"]]
        window = s.get("smooth", 1)
        smoothed = smooth([y for _, y in pts], window) if window > 1 else None
        entry = by_label.setdefault(s["label"], {**s, "points": [], "smoothed": [], "raw": False})
        if entry["points"] and pts[0][0] - entry["points"][-1][0] > gap:
            entry["points"].append((pts[0][0], float("nan")))      # break the line
            entry["smoothed"].append(float("nan"))
        entry["points"] += pts
        entry["smoothed"] += smoothed or [y for _, y in pts]
        entry["raw"] |= smoothed is not None
    return list(by_label.values())


def draw_joined(ax, s: dict, ylim) -> None:
    color = COLORS[s.get("color", 0) % len(COLORS)]
    xs, ys = zip(*s["points"])
    if s.get("style") == "markers":
        ax.plot(xs, ys, linestyle="none", marker="D", markersize=5, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        return
    if s["raw"]:
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)
    ax.plot(xs, s["smoothed"], color=color, linewidth=1.8, zorder=3)
    if ylim and ylim[0] is not None:
        for x, y in s["points"]:
            if y < ylim[0]:
                ax.annotate(f"{y:.3g}", (x, ylim[0]), xytext=(7, 3), textcoords="offset points",
                            fontsize=7.5, color=color, va="bottom")
                ax.plot([x], [ylim[0]], marker="v", color=color, markersize=6, clip_on=False,
                        zorder=5)


def steps_panel(fig, ax, model: dict, panel: dict, first: bool, last: bool) -> None:
    stages = model["stages"]
    offsets = stage_offsets(stages)
    total = offsets[-1]
    style_axis(ax)
    ax.set_xlim(0, total)
    for i, st in enumerate(stages):
        if i % 2:
            ax.axvspan(offsets[i], offsets[i + 1], color=STAGE_TINT[1], zorder=0, linewidth=0)
        if i:
            ax.axvline(offsets[i], color=EDGE, linewidth=1, zorder=1)
    ylim = panel.get("ylim")
    all_y = []
    series = joined_series(panel, offsets)
    for s in series:
        draw_joined(ax, s, ylim)
        all_y += [y for _, y in s["points"]
                  if y == y and not (ylim and ylim[0] is not None and y < ylim[0])]
    if panel.get("zero"):
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=":")
    for ref in panel.get("ref", []):
        c = COLORS[ref["color"] % len(COLORS)] if "color" in ref else MUTED
        ax.axhline(ref["y"], color=c, linewidth=1, linestyle="--", alpha=0.7)
    lo, hi = (ylim or [None, None])
    ys = all_y + [ref["y"] for ref in panel.get("ref", [])] + ([0] if panel.get("zero") else [])
    pad = 0.08 * (max(ys) - min(ys) or 1)
    ax.set_ylim(lo if lo is not None else min(ys) - pad, hi if hi is not None else max(ys) + pad)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(8))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    if not last:
        plt.setp(ax.get_xticklabels(), visible=False)
    measured = {s["stage"] for s in panel["series"] if s["points"]}
    if panel.get("empty"):
        for i in range(len(stages)):
            if i not in measured and stages[i]["steps"] / total > 0.12:
                ax.text((offsets[i] + offsets[i + 1]) / 2, 0.5, panel["empty"],
                        transform=ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=8.5, color=MUTED, style="italic")
    rel = model.get("release")
    if rel:
        x = offsets[rel["stage"]] + rel["step"]
        ax.axvline(x, color=INK, linewidth=1, linestyle="--", zorder=2)
        if first:
            ax.annotate("released", (x, 1), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                        textcoords="offset points", ha="right", va="top", fontsize=7.5, color=INK)
    if first:
        for i, st in enumerate(stages):
            narrow = st["steps"] / total < 0.09
            at_end = narrow and i == len(stages) - 1        # a narrow last stage: label leftwards
            x = offsets[i + 1] if at_end else offsets[i] if narrow else (offsets[i] + offsets[i + 1]) / 2
            ax.annotate(f"{i + 1} {st['name']}", (x, 1),
                        xycoords=("data", "axes fraction"), xytext=(0, 26),
                        textcoords="offset points",
                        ha="right" if at_end else "left" if narrow else "center",
                        va="bottom", fontsize=9.5, color=MUTED, weight="bold")
    title = ax.text(0, 1.04, panel["title"], transform=ax.transAxes, fontsize=10.5,
                    weight="bold", color=INK, va="bottom")
    ax.annotate(panel["ylabel"], (1, 0), xycoords=title, xytext=(8, 0), textcoords="offset points",
                va="bottom", fontsize=9, color=MUTED, annotation_clip=False)
    handles = legend_handles(panel)
    if len(handles) > 1:
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1), ncol=len(handles),
                  frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2,
                  borderaxespad=0.2)
    if last:
        ax.set_xlabel("training steps, all stages end to end", fontsize=9, color=INK)


def other_row(fig, cell, panels: list[dict]) -> None:
    grid = GridSpecFromSubplotSpec(1, len(panels), subplot_spec=cell, wspace=0.32)
    for j, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, j])
        style_axis(ax)
        if panel["type"] == "bars":
            cats, n = panel["categories"], len(panel["series"])
            width = 0.8 / n
            for k, s in enumerate(panel["series"]):
                xs = [i + (k - (n - 1) / 2) * width for i in range(len(cats))]
                ax.bar(xs, s["values"], width=width * 0.92, color=COLORS[s["color"]],
                       label=s["label"])
            ax.set_xticks(range(len(cats)), cats)
            ax.set_ylim(0, max(max(s["values"]) for s in panel["series"]) * 1.45)
            for ref in panel.get("ref", []):
                ax.axhline(ref["y"], color=INK, linewidth=1, linestyle="--", label=ref["label"])
            ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2, handlelength=1.4,
                      columnspacing=0.9)
        else:
            for s in panel["series"]:
                draw_series(ax, s | {"style": "both"}, None)
            ax.set_xlabel(panel["xlabel"], fontsize=9, color=INK)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            ax.legend(handles=legend_handles(panel), fontsize=8, frameon=False, loc="best",
                      handlelength=1.6)
        for ref in (panel.get("ref", []) if panel["type"] != "bars" else []):
            ax.axhline(ref["y"], color=MUTED, linewidth=1, linestyle="--")
            ax.annotate(ref["label"], (1, ref["y"]), xycoords=("axes fraction", "data"),
                        xytext=(-3, 3), textcoords="offset points", fontsize=7.5, color=MUTED,
                        ha="right")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_ylabel(panel["ylabel"], fontsize=9, color=INK)
        ax.text(0, 1.04, panel["title"], transform=ax.transAxes, fontsize=10.5, weight="bold",
                color=INK, va="bottom")


def render(model: dict, out: Path, preview: Path | None) -> None:
    rows = [row for row in model["rows"] if row[0].get("plot", True)]
    steps = [row[0] for row in rows if row[0].get("type", "steps") == "steps"]
    others = [row for row in rows if row[0].get("type", "steps") != "steps"]
    heights = [1.0] * len(steps) + [1.6] * len(others)
    fig = plt.figure(figsize=(10, 1.9 + 1.75 * sum(heights)))
    top = 1 - 1.55 / fig.get_figheight()
    bottom = 0.6 / fig.get_figheight()
    grid = GridSpec(len(heights), 1, figure=fig, height_ratios=heights, left=0.08, right=0.985,
                    top=top, bottom=bottom, hspace=0.5)
    fig.text(0.08, 1 - 0.3 / fig.get_figheight(), model["title"], fontsize=16, weight="bold",
             color=INK, va="center")
    fig.text(0.08, 1 - 0.56 / fig.get_figheight(), model["subtitle"], fontsize=10, color=MUTED,
             va="center")
    first_ax = None
    for i, panel in enumerate(steps):
        ax = fig.add_subplot(grid[i], sharex=first_ax)
        first_ax = first_ax or ax
        steps_panel(fig, ax, model, panel, first=i == 0, last=i == len(steps) - 1)
    for j, row in enumerate(others):
        other_row(fig, grid[len(steps) + j], row)
    path = out / f"{model['id']}.svg"
    fig.savefig(path, metadata={"Date": None, "Title": model["title"],
                               "Description": model["subtitle"]})
    if preview:
        fig.savefig(preview / f"{model['id']}.png", dpi=110)
    plt.close(fig)
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/data/training_curves.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--preview-dir", type=Path, help="also write PNG previews here")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none",
        "svg.hashsalt": "pidgin-training-curves",
    })
    for model in json.loads(args.data.read_text())["models"]:
        render(model, args.out, args.preview_dir)


if __name__ == "__main__":
    main()
