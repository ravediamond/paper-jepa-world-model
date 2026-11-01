#!/usr/bin/env python3
"""paper-jepa-world-model: the two figures, from repro/*/expected/ only.
fig/table2.pdf: Table 2 as a dot plot (rest: AUROC with its 95 % bootstrap CI; anchor: mean of 5 seeds with the seed range).
fig/register.pdf: the register as a 20-row strip (use, what it takes from W1, pre-registration class, verdict)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE.parent / "repro"
OUT = HERE / "fig"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
NEEDS = {"encoder": "#2a78d6", "predictor": "#eb6834", "both": "#1baf7a"}
META = {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "pdf.fonttype": 42, "svg.hashsalt": "0"})


def table2():
    c = json.loads((R / "gaps/expected/controls.json").read_text())
    rest = [("W1 (ours)", "W1 at judge frame (LOSO probe)"), ("colour judge G9c-3", "G9c-3 judge score s (no learning)"),
            ("W2′ DINOv2-S", "W2' at judge frame (LOSO probe)"), ("commanded joints", "joints (commanded) at judge frame (LOSO probe)"),
            ("policy identity", "policy identity (LOSO probe)")]
    anc = {"W1 (ours)": "W1 current at anchor", "W2′ DINOv2-S": "W2' current at anchor",
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
        e = c["gap2"][key]; col = INK if lab.startswith("W1") else MUTED
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


def register():
    t = (R / "register/expected/FEASIBILITY.md").read_text().split("## 2.", 1)[1].split("## 3.", 1)[0]
    rows = [[x.strip() for x in l.split("|")] for l in t.splitlines() if l.startswith("| ") and not l.startswith("| gate")]
    uses = json.loads((R / "register/expected/uses.json").read_text())["rows"]
    assert len(uses) == len(rows)
    plt.rcParams["font.size"] = 6
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ax.set_xlim(0, 8.2); ax.set_ylim(-0.7, len(rows) - 0.3); ax.axis("off")
    for x, h in ((0, "use"), (2.9, "takes from W1"), (4.6, "bar"), (6.3, "verdict")):
        ax.text(x, len(rows) - 0.4, h, fontsize=6.5, color=MUTED, va="bottom", weight="bold")
    for i, (r, u) in enumerate(zip(rows, uses)):
        assert r[1].startswith(u["gate"]), (r[1], u["gate"])
        y = len(rows) - 1 - i
        name = r[1].split(" (")[0].replace("**", "").replace(" on device", "").replace("REAP v1", "REAP v1 / v2")
        v = r[6].replace("**", "").replace(" (retrospective)", " (retro.)").replace(" (policy confound)", "")
        if i % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f3ef", lw=0, zorder=0)
        ax.text(0, y, name[:24], va="center", color=INK)
        ax.plot(2.97, y, "s", ms=4, color=NEEDS[u["needs"]], mec="white", mew=0.5)
        ax.text(3.12, y, u["needs"], va="center", color=INK)
        ax.text(4.6, y, r[7] if r[7] != "—" else "— (not W1)", va="center", color=INK if r[7] == "bar-first" else MUTED)
        ax.text(6.3, y, v[:20], va="center", color=INK, weight="bold" if ("pass" in v.lower() and "fail" not in v.lower()) or "met" in v.lower() else "normal")
    fig.savefig(OUT / "register.pdf", bbox_inches="tight", metadata=META); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); table2(); register(); print("fig/table2.pdf, fig/register.pdf written")
