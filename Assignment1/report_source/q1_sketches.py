"""Typeset the two schematic plots in the handwritten Q1 solution."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parent / "report_figures"
for t in (1, 0):
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    # Schematic geometry from the handwritten sketches; only symbolic ticks.
    ax.plot([0.2, 1, 3, 5], [3.6, 1.45, 0.5, 1.0], color="black", lw=1.8)
    ax.plot([1, 1], [0, 3.7], color="0.5", ls="--", lw=0.8)
    ax.plot([3, 3], [0, 3.7], color="0.5", ls="--", lw=0.8)
    ax.scatter([3], [0.5], facecolors="white", edgecolors="black", s=45, zorder=3)
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 4)
    ax.set_xticks([1, 3], [rf"$1-x_{t}$", rf"$3-x_{t}$"])
    ax.set_yticks([])
    ax.text(0, 4.12, rf"$F(u_{t})$", ha="center", fontsize=12)
    ax.text(5.65, 0, rf"$u_{t}$", va="center", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot(5.5, 0, ">k", clip_on=False, ms=5)
    ax.plot(0, 4, "^k", clip_on=False, ms=5)
    fig.tight_layout(pad=1.1)
    fig.savefig(out / f"q1_stage{t}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
