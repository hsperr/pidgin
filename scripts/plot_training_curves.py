"""Render the documentation's training curves from the committed numeric snapshot.

    python -m pip install matplotlib
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --preview-dir /tmp/pidgin-curves

No models, original research logs, or dashboard server are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "docs/data/training_curves.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--preview-dir", type=Path, help="also write PNG previews here")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#334155", "text.color": "#172634",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "svg.fonttype": "none", "svg.hashsalt": "pidgin-training-curves",
    })
    for model in json.loads(args.data.read_text())["models"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.69, wspace=0.38)
        fig.text(0.055, 0.95, model["title"], fontsize=18, weight="bold")
        fig.text(0.055, 0.89, model["subtitle"], fontsize=11)
        fig.text(0.055, 0.825, "Dashed line: released checkpoint · logged values, no added smoothing",
                 fontsize=9, color="#64748b")
        for ax, panel, color in zip(axes, model["panels"], ["#176b80", "#925430"]):
            x, y = zip(*panel["points"])
            ax.plot(x, y, color=color, linewidth=1.7,
                    marker="o" if len(x) < 35 else None, markersize=3)
            ax.axvline(model["released_step"], color="#7c8796", linestyle="--", linewidth=1)
            ax.set(title=panel["title"], xlabel="Training step", ylabel=panel["ylabel"])
            ax.title.set_fontsize(11)
            ax.set_xlim(0, model["released_step"] * 1.04)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
            ax.set_axisbelow(True)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            for side in ["bottom", "left"]:
                ax.spines[side].set_color("#cbd5e1")
        path = args.out / f"{model['id']}.svg"
        fig.savefig(path, metadata={"Date": None, "Title": model["title"],
                                   "Description": model["subtitle"]})
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"{model['id']}.png", dpi=150)
        plt.close(fig)
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
