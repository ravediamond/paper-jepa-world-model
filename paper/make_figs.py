#!/usr/bin/env python3
"""paper-jepa-world-model: the figure, from repro/*/expected/ only.
fig/table2.pdf: the controls table as a dot plot (rest: AUROC with its 95 % bootstrap CI; anchor: mean of 5 seeds with the seed range)."""
import json
from pathlib import Path

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "style"))
import shisso_colors as sc
sc.use()
R = HERE.parent / "repro"
OUT = HERE / "fig"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
META = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42, "svg.hashsalt": "0"})


def table2():
    c = json.loads((R / "gaps/expected/controls.json").read_text())
    rest = [("our world model", "W1 at judge frame (LOSO probe)"), ("colour judge (no learning)", "G9c-3 judge score s (no learning)"),
            ("frozen DINOv2-S", "W2' at judge frame (LOSO probe)"), ("commanded joints", "joints (commanded) at judge frame (LOSO probe)"),
            ("policy identity", "policy identity (LOSO probe)")]
    anc = {"our world model": "W1 current at anchor", "frozen DINOv2-S": "W2' current at anchor",
           "commanded joints": "joints (commanded) current at anchor"}
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.55), sharey=True, gridspec_kw={"wspace": 0.08})
    ys = list(range(len(rest)))[::-1]
    for ax, title in zip(axes, ("at rest (95 % CI)", f"grasp anchor (seed range, n = {c['w1c1']['scored']})")):
        ax.set_title(title, fontsize=7, color=INK, loc="left")
        ax.set_xlim(0.45, 1.02); ax.axvline(0.5, color=GRID, lw=1, zorder=0)
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for y, (lab, key) in zip(ys, rest):
        e = c["gap2"][key]; col = sc.ACCENT if lab.startswith("our") else sc.GREY
        axes[0].plot(e["ci95"], [y, y], color=col, lw=1.2, solid_capstyle="round")
        axes[0].plot(e["auroc"], y, "o", ms=4, color=col, mec="white", mew=0.8)
        if lab in anc:
            a = c["w1c1"][anc[lab]]; s = a["auroc_oof_per_seed"]
            axes[1].plot([min(s), max(s)], [y, y], color=col, lw=1.2, solid_capstyle="round")
            axes[1].plot(a["auroc"], y, "o", ms=4, color=col, mec="white", mew=0.8)
        else:
            axes[1].text(0.47, y, "n/a", va="center", color=MUTED, fontsize=6)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([l for l, _ in rest])
    for ax in axes:
        ax.set_xlabel("outcome AUROC")
    fig.savefig(OUT / "table2.pdf", bbox_inches="tight", metadata=META); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); table2(); print("fig/table2.pdf written")
