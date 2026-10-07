"""Render the documentation's training curves from the committed numeric snapshot.

    python -m pip install matplotlib
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --pdf-dir paper/figures --preview-dir /tmp/curves

Reads docs/data/training_curves.json and writes one SVG per measurement to
docs/figures/<model>/<measurement>.svg. A figure has no title; the document around it
names it. "steps" panels (the default) put the stages that measured the value end to
end on one step axis, with a line and a label at each stage change. "xy" and "bars"
panels are single measurements of the released model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#1f6fb2", "#d9661f", "#2e8b57", "#7b4fc4", "#c0392b", "#0f8a8a"]
INK, MUTED, GRID, EDGE = "#1e293b", "#64748b", "#e2e8f0", "#cbd5e1"
STAGE_TINT = "#eef2f7"
SIZE = (6.4, 3.0)                                  # inches


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


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


def color_of(s: dict) -> str:
    return COLORS[s.get("color", 0) % len(COLORS)]


def legend_handles(panel: dict) -> list:
    seen, handles = set(), []
    for s in panel.get("series", []):
        if s["label"] in seen:
            continue
        seen.add(s["label"])
        if s.get("style") == "markers":
            handles.append(plt.Line2D([], [], linestyle="none", marker="D", markersize=5,
                                      color=color_of(s), label=s["label"]))
        else:
            handles.append(plt.Line2D([], [], color=color_of(s), linewidth=1.8, label=s["label"]))
    if panel.get("ref_label"):
        handles.append(plt.Line2D([], [], color=MUTED, linewidth=1, linestyle="--",
                                  label=panel["ref_label"]))
    return handles


def legend_below(ax, handles: list, labels: list[str] | None = None) -> None:
    """A legend under the x-axis label, so the space above the plot holds the stage names."""
    if len(handles) > 1:
        labels = labels or [h.get_label() for h in handles]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(handles),
                  frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.4)


def stage_offsets(stages: list[dict]) -> list[int]:
    out, total = [], 0
    for st in stages:
        out.append(total)
        total += st["steps"]
    return out + [total]


def joined_series(panel: dict, offsets: list[int], gap: float) -> list[dict]:
    """One series per label, its stages laid end to end on the shared step axis."""
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
    color = color_of(s)
    xs, ys = zip(*s["points"])
    if s.get("style") == "markers":
        ax.plot(xs, ys, linestyle="none", marker="D", markersize=5, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        return
    if s["raw"]:
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.25)
    marker = "o" if s.get("style") == "both" and len(xs) <= 40 else None
    ax.plot(xs, s["smoothed"], color=color, linewidth=1.8, marker=marker, markersize=3, zorder=3)
    if ylim and ylim[0] is not None:
        for x, y in s["points"]:
            if y < ylim[0]:
                ax.annotate(f"{y:.3g}", (x, ylim[0]), xytext=(7, 3), textcoords="offset points",
                            fontsize=7.5, color=color, va="bottom")
                ax.plot([x], [ylim[0]], marker="v", color=color, markersize=6, clip_on=False,
                        zorder=5)


def steps_figure(model: dict, panel: dict):
    stages = model["stages"]
    measured = sorted({s["stage"] for s in panel["series"] if s["points"]})
    first, last = measured[0], measured[-1]
    offsets = stage_offsets(stages)
    shown = range(first, last + 1)
    x0, x1 = offsets[first], offsets[last + 1]
    fig, ax = plt.subplots(figsize=SIZE)
    style_axis(ax)
    ax.set_xlim(0, x1 - x0)
    shift = [o - x0 for o in offsets]
    for i in shown:
        if (i - first) % 2:
            ax.axvspan(shift[i], shift[i + 1], color=STAGE_TINT, zorder=0, linewidth=0)
        if i > first:
            ax.axvline(shift[i], color=EDGE, linewidth=1, zorder=1)
    ylim = panel.get("ylim")
    all_y = []
    for s in joined_series(panel, shift, 0.05 * (x1 - x0)):
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
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_ylabel(panel["ylabel"], fontsize=9, color=INK)
    if len(shown) > 1:
        ax.set_xlabel("training steps, stages end to end", fontsize=9, color=INK)
        width = x1 - x0
        for i in shown:
            narrow = stages[i]["steps"] / width < 0.12
            at_end = narrow and i == last
            x = shift[i + 1] if at_end else shift[i] if narrow else (shift[i] + shift[i + 1]) / 2
            ax.annotate(stages[i]["name"], (x, 1), xycoords=("data", "axes fraction"),
                        xytext=(0, 5),
                        textcoords="offset points", fontsize=8, color=MUTED, weight="bold",
                        ha="right" if at_end else "left" if narrow else "center", va="bottom")
    else:
        ax.set_xlabel(f"training steps ({stages[first]['name'].lower()} stage)", fontsize=9,
                      color=INK)
    rel = model.get("release")
    if rel and first <= rel["stage"] <= last:
        x = shift[rel["stage"]] + rel["step"]
        ax.axvline(x, color=INK, linewidth=1, linestyle="--", zorder=2)
        ax.annotate("released", (x, 1), xycoords=("data", "axes fraction"), xytext=(-3, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=7.5, color=INK)
    legend_below(ax, legend_handles(panel))
    return fig


def single_figure(panel: dict):
    fig, ax = plt.subplots(figsize=SIZE)
    style_axis(ax)
    if panel["type"] == "bars":
        cats, n = panel["categories"], len(panel["series"])
        width = 0.8 / n
        for k, s in enumerate(panel["series"]):
            xs = [i + (k - (n - 1) / 2) * width for i in range(len(cats))]
            ax.bar(xs, s["values"], width=width * 0.92, color=color_of(s))
        ax.set_xticks(range(len(cats)), cats)
        ax.set_ylim(0, max(max(s["values"]) for s in panel["series"]) * 1.15)
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_of(s)) for s in panel["series"]]
        handles += [plt.Line2D([], [], color=MUTED, linewidth=1, linestyle="--")
                    for _ in panel.get("ref", [])]
        labels = [s["label"] for s in panel["series"]] + [r["label"] for r in panel.get("ref", [])]
        legend_below(ax, handles, labels)
    else:
        for s in panel["series"]:
            xs, ys = zip(*s["points"])
            ax.plot(xs, ys, color=color_of(s), linewidth=1.8, marker="o", markersize=3)
        ax.set_xlabel(panel["xlabel"], fontsize=9, color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        legend_below(ax, legend_handles(panel))
    for ref in panel.get("ref", []):
        ax.axhline(ref["y"], color=MUTED, linewidth=1, linestyle="--")
        if panel["type"] != "bars":
            ax.annotate(ref["label"], (1, ref["y"]), xycoords=("axes fraction", "data"),
                        xytext=(-3, 3), textcoords="offset points", fontsize=7.5, color=MUTED,
                        ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel(panel["ylabel"], fontsize=9, color=INK)
    return fig


def render(model: dict, out: Path, pdf: Path | None, preview: Path | None) -> None:
    name = model["id"].removesuffix("_training")
    for row in model["rows"]:
        for panel in row:
            kind = panel.get("type", "steps")
            fig = steps_figure(model, panel) if kind == "steps" else single_figure(panel)
            stem = panel.get("file", slug(panel["title"]))
            for folder, ext, kw in ((out, "svg", {"metadata": {"Date": None}}),
                                    (pdf, "pdf", {"metadata": {"CreationDate": None}}),
                                    (preview, "png", {"dpi": 110})):
                if folder:
                    (folder / name).mkdir(parents=True, exist_ok=True)
                    fig.savefig(folder / name / f"{stem}.{ext}", bbox_inches="tight",
                                pad_inches=0.08, **kw)
            plt.close(fig)
            path = out / name / f"{stem}.svg"
            print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/data/training_curves.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--pdf-dir", type=Path, help="also write PDFs here (for a paper)")
    parser.add_argument("--preview-dir", type=Path, help="also write PNG previews here")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "svg.fonttype": "none",
        "svg.hashsalt": "pidgin-training-curves",
    })
    for model in json.loads(args.data.read_text())["models"]:
        render(model, args.out, args.pdf_dir, args.preview_dir)


if __name__ == "__main__":
    main()
