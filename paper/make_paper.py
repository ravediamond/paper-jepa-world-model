#!/usr/bin/env python3
"""paper-jepa-world-model: fill paper/DRAFT_4p.md.in from repro/*/expected/ into numbers.json, DRAFT_4p.md and main.tex.
Every number has an anchor; a row literal that is not found in its source file fails the build."""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / "repro"
G = R / "gaps/expected"
REG = R / "register/expected"
N, A = {}, {}


def put(key, value, anchor):
    N[key] = value; A[key] = anchor


def row(gate):
    """The FEASIBILITY.md §2 row whose first cell starts with `gate`."""
    t = (REG / "FEASIBILITY.md").read_text()
    for line in t.splitlines():
        if line.startswith("| " + gate + " "):
            return line
    raise SystemExit(f"row not found: {gate}")


def lit(key, gate, text, value=None):
    assert text in row(gate), f"{text!r} not in the {gate} row"
    put(key, value if value is not None else text, f"repro/register/expected/FEASIBILITY.md §2 row {gate}: '{text}'")


def lit_file(key, rel, text, value):
    assert text in (R / rel).read_text(), f"{text!r} not in {rel}"
    put(key, value, f"repro/{rel}: '{text}'")


def f4(x):
    return f"{x:.4f}"


def main():
    c = json.loads((G / "controls.json").read_text())
    for k in ("n", "success", "failure", "sessions"):
        put(k, str(c[k]), f"repro/gaps/expected/controls.json {k}")
    names = {"W1": "W1 at judge frame (LOSO probe)", "W2p": "W2' at judge frame (LOSO probe)",
             "joints": "joints (commanded) at judge frame (LOSO probe)", "policy": "policy identity (LOSO probe)",
             "judge": "G9c-3 judge score s (no learning)"}
    for k, name in names.items():
        e = c["gap2"][name]; a = f"repro/gaps/expected/controls.json gap2/{name}"
        put(f"rest_{k}", f4(e["auroc"]), a + "/auroc")
        put(f"rest_{k}_lo", f4(e["ci95"][0]), a + "/ci95[0]"); put(f"rest_{k}_hi", f4(e["ci95"][1]), a + "/ci95[1]")
    w = c["w1c1"]
    put("anc_scored", str(w["scored"]), "repro/gaps/expected/controls.json w1c1/scored")
    put("anc_fail", str(w["failures"]), "repro/gaps/expected/controls.json w1c1/failures")
    for k, name in {"W1": "W1 current at anchor", "W2p": "W2' current at anchor", "joints": "joints (commanded) current at anchor"}.items():
        put(f"anc_{k}", f4(w[name]["auroc"]), f"repro/gaps/expected/controls.json w1c1/{name}/auroc")
    ci = json.loads((G / "anchor_ci_mean_of_seeds.json").read_text())["features"]
    se = json.loads((G / "anchor_ci_sensitivity.json").read_text())["features"]
    for k in ("W1", "W2p", "joints"):
        a = f"repro/gaps/expected/anchor_ci_mean_of_seeds.json features/{k}/ci95_mean_of_seeds"
        put(f"anc_{k}_lo", f4(ci[k]["ci95_mean_of_seeds"][0]), a + "[0]"); put(f"anc_{k}_hi", f4(ci[k]["ci95_mean_of_seeds"][1]), a + "[1]")
        a = f"repro/gaps/expected/anchor_ci_sensitivity.json features/{k}/"
        put(f"anc_{k}_avg", f4(se[k]["auroc_seed_averaged"]), a + "auroc_seed_averaged")
        put(f"anc_{k}_s0", f4(se[k]["auroc_seed0"]), a + "auroc_seed0")
    for k, name in {"W1": "W1", "W2p": "W2'", "joints": "joints (commanded)"}.items():
        put(f"surv_tau_{k}", f4(c["surveyor_gate0"][name]["tau"]), f"repro/gaps/expected/controls.json surveyor_gate0/{name}/tau")
    ks = json.loads((G / "vla_jepa_keyset.json").read_text())
    put("wm_keys", str(ks["world_model_keys"]), "repro/gaps/expected/vla_jepa_keyset.json world_model_keys")
    put("wm_missing", str(ks["world_model_keys_missing"]), "repro/gaps/expected/vla_jepa_keyset.json world_model_keys_missing")
    sp = json.loads((REG / "split.json").read_text())
    put("train_eps", str(sp["n_train"]), "repro/register/expected/split.json n_train")
    put("val_eps", str(sp["n_val"]), "repro/register/expected/split.json n_val")
    put("teleop_eps", str(sp["n_train"] + sp["n_val"]), "repro/register/expected/split.json n_train + n_val")
    lit_file("audit_dropped", "register/expected/AUDIT_W1_LOADS.md", "192 encoder tensors dropped", "192")
    lit_file("audit_tf", "register/expected/AUDIT_W1_LOADS.md", "transformers 5.5.4", "5.5.4")

    rows = [l for l in (REG / "FEASIBILITY.md").read_text().split("## 2.", 1)[1].split("## 3.", 1)[0].splitlines()
            if l.startswith("| ") and not l.startswith("| gate")]
    cnt = {"pass": 0, "met": 0, "np": 0, "fail": 0}; pre = {"bar-first": 0, "same-commit": 0, "—": 0}
    for l in rows:
        cells = [x.strip() for x in l.split("|")]
        v, p = cells[6].lower(), cells[7]
        cnt["np" if "not proven" in v else "met" if "met" in v else "pass" if "pass" in v and "fail" not in v else "fail"] += 1
        if p in pre: pre[p] += 1
    a = "repro/register/expected/FEASIBILITY.md §2 table, counted by make_paper.py"
    put("rows", str(len(rows)), a)
    for k in cnt: put(f"n_{k}", str(cnt[k]), a + f" (verdict {k})")
    pa = json.loads((REG / "prereg_audited.json").read_text())
    assert [r["register_gate"] for r in pa["rows"]] == [l.split("|")[1].strip()[:len(r["register_gate"])] for l, r in zip(rows, pa["rows"])]
    a = "repro/register/expected/prereg_audited.json rows, audited class, counted by make_paper.py"
    cls = [r["class"] for r in pa["rows"]]
    put("prereg_matched", str(len(pa["rows"]) - len(pa["unmatched"])), a + " (rows minus unmatched)")
    put("bar_first", str(cls.count("bar-first")), a + " (bar-first)"); put("same_commit", str(cls.count("same-commit")), a + " (same-commit)")
    put("same_commit_gates", ", ".join(r["register_gate"] for r in pa["rows"] if r["class"] == "same-commit"), a + " (same-commit gates)")
    vs = [l.split("|")[6].lower() for l in rows]
    passes = [c for v, c in zip(vs, cls) if "pass" in v and "fail" not in v and "not proven" not in v]
    put("pass_bar_first", str(passes.count("bar-first")), a + " (bar-first among the verdict-pass rows of FEASIBILITY.md §2)")
    put("prereg_v1_bar_first", str(pre["bar-first"]), "repro/register/expected/FEASIBILITY.md §2 table, counted by make_paper.py (pre-reg. bar-first, commit order only)")
    put("prereg_v1_same_commit", str(pre["same-commit"]), "repro/register/expected/FEASIBILITY.md §2 table, counted by make_paper.py (pre-reg. same-commit, commit order only)")
    uses = json.loads((REG / "uses.json").read_text())["rows"]
    for k in ("encoder", "predictor", "both"):
        put(f"uses_{k}", str(sum(u["needs"] == k for u in uses)), f"repro/register/expected/uses.json rows, needs = {k}")

    g5 = json.loads((G / "gap5_pooled_rerun.json").read_text()); a = "repro/gaps/expected/gap5_pooled_rerun.json"
    put("g5_w1", f4(g5["terms"]["W1"]), a + " terms/W1"); put("g5_rand", f4(g5["terms"]["W1_random_init"]), a + " terms/W1_random_init")
    put("g5_val", str(g5["val_episodes"]), a + " val_episodes")
    d = g5["pairs"]["W1_minus_W1_random_init"]
    put("g5_diff", f4(d["point"]).replace("-", "−"), a + " pairs/W1_minus_W1_random_init/point")
    put("g5_lo", f4(d["ci95"][0]).replace("-", "−"), a + " pairs/W1_minus_W1_random_init/ci95[0]")
    put("g5_hi", "+" + f4(d["ci95"][1]), a + " pairs/W1_minus_W1_random_init/ci95[1]")
    lit_file("g5_unstable", "gaps/expected/gap5_floor_stability.md", "It was 1.4383 on the first", "1.4383")
    lit_file("g5_orin", "gaps/expected/gap5_floor_stability.md", "0.8042 on the Orin", "0.8042")
    m1 = json.loads((G / "M1_diag_g0_repro.json").read_text()); a = "repro/gaps/expected/M1_diag_g0_repro.json"
    for k, name in {"xep": "cross_episode_ratio", "cf": "constant_first_ratio"}.items():
        put(f"m1_{k}", f"{m1['by_k']['30'][name]:.3f}", a + f" by_k/30/{name}")
        put(f"m1_{k}_g0", f"{m1['g0_k30'][name]:.3f}", a + f" g0_k30/{name}")
    lit_file("m1_auroc", "gaps/expected/M1_diag_note.md", "AUROC 0.775 < 0.9", "0.775")
    m3 = json.loads((G / "M3_diag_action_shuffle.json").read_text()); a = "repro/gaps/expected/M3_diag_action_shuffle.json"
    pct = lambda x, y: 100 * (x - y) / y
    tok = [100 * r["token_rel_l2"] for r in m3["rows"]]; zero = [abs(pct(r["zero_tokens"], r["true_1"])) for r in m3["rows"]]
    for i, r in enumerate(m3["rows"]):
        b = a + f" rows[{i}]"
        put(f"m3_{i}_ep", str(r["episode"]), b + "/episode"); put(f"m3_{i}_fr", str(r["anchor"]), b + "/anchor")
        put(f"m3_{i}_true", f4(r["true_1"]), b + "/true_1")
        put(f"m3_{i}_sh", f"{pct(r['shuffled'], r['true_1']):+.3f}".replace("-", "−"), b + " 100 (shuffled − true_1) / true_1")
        put(f"m3_{i}_tok", f"{tok[i]:.1f}", b + " 100 token_rel_l2")
        put(f"m3_{i}_zero", f"{pct(r['zero_tokens'], r['true_1']):+.2f}".replace("-", "−"), b + " 100 (zero_tokens − true_1) / true_1")
    put("m3_n", str(len(m3["rows"])), a + " len(rows)")
    put("m3_tok_lo", f"{min(tok):.0f}", a + " min over rows of 100 token_rel_l2"); put("m3_tok_hi", f"{max(tok):.0f}", a + " max over rows of 100 token_rel_l2")
    put("m3_sh", f"{100 * m3['mean_abs_diff_shuffled'] / m3['mean_true']:.2f}", a + " 100 mean_abs_diff_shuffled / mean_true")
    put("m3_zero", f"{sum(zero) / len(zero):.2f}", a + " mean over rows of 100 |zero_tokens − true_1| / true_1")
    put("m3_zero_lo", f"{min(zero):.1f}", a + " min over rows of 100 |zero_tokens − true_1| / true_1")
    put("m3_zero_hi", f"{max(zero):.1f}", a + " max over rows of 100 |zero_tokens − true_1| / true_1")
    lit_file("m3_auroc", "gaps/expected/M3_diag_note.md", "class-5 AUROC is 0.501 against the 0.9", "0.501")

    lit("g0_err", "G0", "0.31"); lit("g0_copy", "G0", "0.89")
    lit("g1b_top1", "G1b", "0.60"); lit("g1b_actions", "G1b", "0.68–0.90"); lit("g1b_bar", "G1b", "≥ 0.8", "0.8")
    lit("w1a_hits", "W1-A", "2/8 at 2/42")
    lit("c1_imagined", "W1-C1", "imagined 0.70", "0.70"); lit("c1_current", "W1-C1", "current 0.82", "0.82")
    lit("c1_bar", "W1-C1", "≥ 0.85, ≥ +0.05")
    lit("lejepa_w1", "LeJEPA", "R² 0.565", "0.565"); lit("lejepa_w2p", "LeJEPA", "W2′ 0.867", "0.867")
    lit("surv_paper", "SURVEYOR", "paper 0.20", "0.20")
    lit("g10_rest", "G10", "W1 1.000", "1.000"); lit("g10_anticip", "G10", "+0.179")
    lit_file("anc_lead", "gaps/scripts/run_w1c_offline.py", "ANCHOR_BEFORE_CLOSE, K_IMAGINE = 15, 30", "15")
    lit_file("g10_n", "gaps/expected/README.md", "against judge labels on 87", "87")
    lit_file("sai_eps", "register/expected/FEASIBILITY.md", "10,030 LIBERO-Plus", "10,030")
    lit_file("sai_gain", "register/expected/FEASIBILITY.md", "+0.03 points", "+0.03")
    lit_file("cost_w1", "register/expected/FEASIBILITY.md", "3.30) predicts", "3.30")
    lit_file("arm_cost", "register/expected/FEASIBILITY.md", "on a $200 robot", "200")
    lit_file("scale_eps", "register/expected/FEASIBILITY.md", "(653 episodes", "653")
    lit_file("audit_trained", "register/expected/AUDIT_W1_LOADS.md", "**Trained: 26 rows**", "26")
    lit_file("audit_random", "register/expected/AUDIT_W1_LOADS.md", "**random: 1**", "1")

    # data-scale probes (repro/data-scale): one model per scale, the same probes, verdicts recomputed by repro/data-scale/l0_check.py
    DS = R / "data-scale"
    cv = json.loads((DS / "expected/curve.json").read_text())
    n_eps = {"S1": 45, "S3": 157, "S4": 810, "S5a": 2909}
    for k, s in (("1", "S1"), ("3", "S3"), ("4", "S4"), ("5", "S5a")):
        v = json.loads((DS / f"expected/{s}/verdict.json").read_text()); a = f"repro/data-scale/expected/{s}/verdict.json "
        put(f"sc{k}_eps", f"{n_eps[s]:,}", f"repro/data-scale/README.md scale table ({s}); list lengths checked by l0_check.py")
        put(f"sc{k}_frames", f"{cv[s]['train_frames']:,}", "repro/data-scale/expected/curve.json train_frames")
        put(f"sc{k}_r2", f"{v['P1']['mean_r2']:.3f}", a + "P1/mean_r2"); put(f"sc{k}_aniso", f"{v['P1']['anisotropy']:.3f}", a + "P1/anisotropy")
        c30 = v["P2"]["cells"]["d30"]
        put(f"sc{k}_top1", f"{v['P2']['best_top1']:.3f}", a + "P2/best_top1")
        put(f"sc{k}_d30", f"{c30['model_top1']:.3f}", a + "P2/cells/d30/model_top1")
        put(f"sc{k}_gap", f"{c30['diff']:.3f}", a + "P2/cells/d30/diff")
        put(f"sc{k}_gap_lo", f"{c30['diff_ci95'][0]:.3f}", a + "P2/cells/d30/diff_ci95[0]"); put(f"sc{k}_gap_hi", f"{c30['diff_ci95'][1]:.3f}", a + "P2/cells/d30/diff_ci95[1]")
        b = v["P3"]["bar"]
        put(f"sc{k}_p3", f"{b['arm_ii_minus_max_i_iii']:.2f}", a + "P3/bar/arm_ii_minus_max_i_iii"); put(f"sc{k}_p3v", f"{b['arm_v_minus_iv']['value']:.3f}", a + "P3/bar/arm_v_minus_iv/value")
        put(f"sc{k}_p3w1", f"{b['w1_arm_ii_minus_i']:.2f}", a + "P3/bar/w1_arm_ii_minus_i")
        put(f"sc{k}_p4", f"{v['P4']['g16_set']['l2']['caught_failed']:.0f}", a + "P4/g16_set/l2/caught_failed")
        p5 = v["P5"]
        put(f"sc{k}_p5", f"{p5['auroc']:.3f}", a + "P5/auroc"); put(f"sc{k}_p5_lo", f"{p5['ci95'][0]:.3f}", a + "P5/ci95[0]"); put(f"sc{k}_p5_hi", f"{p5['ci95'][1]:.3f}", a + "P5/ci95[1]")
        put(f"sc{k}_p5d", f"{p5['diff_vs_judge_valid']:+.3f}", a + "P5/diff_vs_judge_valid")
    s3 = json.loads((DS / "expected/S3/verdict.json").read_text())
    put("sc_floor30", f"{s3['P2']['cells']['d30']['action_only_top1']:.3f}", "repro/data-scale/expected/S3/verdict.json P2/cells/d30/action_only_top1 (the same ranker and anchors at every scale)")
    put("sc_random_gain", f"{s3['P3']['bar']['random_init_arm_ii_minus_i']:.2f}", "repro/data-scale/expected/S3/verdict.json P3/bar/random_init_arm_ii_minus_i (one untrained encoder at every scale)")
    put("sc_p3_floor", f"{min(json.loads((DS / f'expected/{s}/verdict.json').read_text())['P3']['bar']['arm_v_minus_iv']['value'] for s in n_eps):.3f}", "repro/data-scale/expected/*/verdict.json P3/bar/arm_v_minus_iv/value, minimum over the scales")
    put("sc_p3_ceil", f"{max(json.loads((DS / f'expected/{s}/verdict.json').read_text())['P3']['bar']['arm_v_minus_iv']['value'] for s in n_eps):.3f}", "repro/data-scale/expected/*/verdict.json P3/bar/arm_v_minus_iv/value, maximum over the scales")
    import math
    put("sc_p3_min", f"{math.floor(100 * min(json.loads((DS / f'expected/{s}/verdict.json').read_text())['P3']['bar']['arm_ii_minus_max_i_iii'] for s in n_eps)) / 100:.2f}", "repro/data-scale/expected/*/verdict.json P3/bar/arm_ii_minus_max_i_iii, minimum over the scales, rounded down to 2 decimals")
    put("sc_judge", f"{s3['P5']['judge_auroc_valid']:.3f}", "repro/data-scale/expected/S3/verdict.json P5/judge_auroc_valid")
    put("sc_steps", f"{json.loads((DS / 'expected/S1/training_end.json').read_text())['step']:,}", "repro/data-scale/expected/S1/training_end.json step")

    assert N["same_commit_gates"] == "G6b", "prose names the same-commit row as the two-level hierarchical progress use"
    put("rest_W1_2", f"{c['gap2'][names['W1']]['auroc']:.2f}", f"repro/gaps/expected/controls.json gap2/{names['W1']}/auroc, 2 decimals")
    put("rest_judge_2", f"{c['gap2'][names['judge']]['auroc']:.2f}", f"repro/gaps/expected/controls.json gap2/{names['judge']}/auroc, 2 decimals")

    tpl = (HERE / "DRAFT_4p.md.in").read_text()
    src = re.sub(r"\{\{(\w+)\}\}", lambda m: N[m.group(1)], tpl).replace("<!-- register -->", register_md(rows, uses, pa["rows"]))
    assert "{{" not in src
    bib = read_bib()
    (HERE / "DRAFT_4p.md").write_text(to_md(src, bib))
    (HERE / "numbers.json").write_text(json.dumps({k: {"value": N[k], "anchor": A[k]} for k in sorted(N)}, indent=1, ensure_ascii=False) + "\n")
    (HERE / "main.tex").write_text(to_tex(src, bib))
    import make_figs; make_figs.OUT.mkdir(exist_ok=True); make_figs.table2(); design_figures()
    print(f"{len(N)} numbers; DRAFT_4p.md, main.tex, numbers.json written")


def design_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import sys
    sys.path.insert(0, str(HERE / "style")); import shisso_colors as sc; sc.use()
    FIGD = HERE / "fig"; FIGD.mkdir(exist_ok=True)
    GREEN, RED = sc.CATEGORICAL[2], sc.CATEGORICAL[3]
    fig, ax = plt.subplots(figsize=(7.0, 1.85)); ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
    lanes = [(12.0, "the encoder: what the world model reads", "last-frame latent, linear probe\non {} rollouts it never saw".format(N["n"]),
              "outcome AUROC {} (colour judge {},\nfrozen DINOv2-S {}, joints {})".format(N["rest_W1"], N["rest_judge"], N["rest_W2p"], N["rest_joints"]), GREEN,
              "45 to {} episodes: {} to {}\nlevel with the judge at every size".format(N["sc5_eps"], N["sc1_p5"], N["sc5_p5"]), GREEN),
             (1.0, "the predictor: what it forecasts", "imagined latents from the current\nlatent and the commanded actions",
              "chunk ranking {} of anchors (bar {}), commanded actions\nalone {}; surprise monitor catches {} failures with {} false alarms".format(N["g1b_top1"], N["g1b_bar"], N["g1b_actions"], N["w1a_hits"].split(" at ")[0], N["w1a_hits"].split(" at ")[1]), RED,
              "45 to {} episodes: best cell {}, {},\n{}, {}; never above the floor".format(N["sc5_eps"], N["sc1_top1"], N["sc3_top1"], N["sc4_top1"], N["sc5_top1"]), RED)]
    cols = [(1.0, 24.0), (30.0, 36.0), (72.0, 27.0)]
    for y, head, a, b, cb, c, cc in lanes:
        for (x, w), txt, col in zip(cols, [a, b, c], ["#8A90A0", cb, cc]):
            ax.add_patch(FancyBboxPatch((x, y), w, 7.4, boxstyle="round,pad=0,rounding_size=1.0", fc="white", ec=col, lw=1.7))
            ax.text(x + w / 2, y + 3.7, txt, ha="center", va="center", fontsize=5.6, linespacing=1.2)
        ax.text(1.0, y + 8.6, head, fontsize=6.3, fontweight="bold", color=sc.ACCENT)
        for x0, x1 in [(25.0, 30.0), (66.0, 72.0)]:
            ax.add_patch(FancyArrowPatch((x0, y + 3.7), (x1, y + 3.7), arrowstyle="-|>", mutation_scale=6, lw=1.0, color=sc.ACCENT))
    for i, (lab, c) in enumerate([("reads the outcome, level with the judge", GREEN), ("misses its threshold", RED)]):
        lx = 62 + i * 21.0; ax.add_patch(FancyBboxPatch((lx, 20.4), 1.4, 1.4, boxstyle="round,pad=0", fc=c, ec="none")); ax.text(lx + 2.1, 21.1, lab, fontsize=5.2, va="center")
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(FIGD / "encoder_predictor.pdf"); fig.savefig(FIGD / "encoder_predictor.png", dpi=200); plt.close(fig)

    # the chain
    fig, ax = plt.subplots(figsize=(7.0, 1.05)); ax.set_xlim(0, 100); ax.set_ylim(0, 15); ax.axis("off")
    steps = [("1  bar committed", "protocol commit\nbefore any number"), ("2  run", "rows stored in\nrepro/<exp>/expected"), ("3  result committed", "result commit\nafter the bar"),
             ("4  L0 recompute", "make l0: every number\nfrom the stored rows"), ("5  paper", "numbers.json: value\nand its anchor")]
    bw, gap = 16.2, 4.0
    for i, (a, b) in enumerate(steps):
        xx = 1 + i * (bw + gap); ax.add_patch(FancyBboxPatch((xx, 3), bw, 10, boxstyle="round,pad=0,rounding_size=1.2", fc="#EDF2FB" if i != 4 else sc.ACCENT, ec=sc.ACCENT, lw=1.0))
        ax.text(xx + bw / 2, 10.3, a, ha="center", va="center", fontsize=6.4, fontweight="bold", color="white" if i == 4 else sc.ACCENT)
        ax.text(xx + bw / 2, 6.0, b, ha="center", va="center", fontsize=5.6, color="white" if i == 4 else "black", linespacing=1.15)
        if i < 4:
            ax.add_patch(FancyArrowPatch((xx + bw, 8), (xx + bw + gap, 8), arrowstyle="-|>", mutation_scale=6, lw=1.0, color=sc.ACCENT))
    ax.text(50, 0.7, "make check regenerates the paper from the stored rows and fails on any difference", ha="center", fontsize=5.8, color=sc.DARKGREY, style="italic")
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(FIGD / "number_chain.pdf"); fig.savefig(FIGD / "number_chain.png", dpi=200); plt.close(fig)


# register table: one row per FEASIBILITY.md §2 row; names and citations from register_names.json, the rest verbatim
PLAIN = [("W2 (frozen DINOv2-S)", "DINOv2-S"), ("W2′", "DINOv2-S"), ("W2", "DINOv2-S"), ("random-init W1", "random-init ours"),
         ("W1", "ours"), ("G1b's", "the ranking row's"), ("v1 ", "first version "), ("v2 ", "second version ")]
BARFIRST = {"bar-first": "yes", "same-commit": "same commit"}


def plain(s):
    for a, b in PLAIN:
        s = s.replace(a, b)
    return s


def register_md(rows, uses, pa):
    names = json.loads((HERE / "register_names.json").read_text())["rows"]
    assert len(names) == len(rows) == len(uses) == len(pa)
    out = ["Table @tab:register: The register: every use of our world model re-tested here (with the frozen-DINOv2 and RoboJEPA rows beside it), what it takes from the model, whether its bar came before the result, and the verdict. Read the takes column against the verdict: the predictor uses fail. Bar, result and verdict are quoted from repro/register/expected/FEASIBILITY.md §2.",
           "", "| use | re-tests | takes | bar first | bar | result | verdict | code |", "|---|---|---|---|---|---|---|---|"]
    for l, u, q, nm in zip(rows, uses, pa, names):
        cells = [x.strip() for x in l.split("|")]
        assert cells[1].startswith(nm["gate"]) and cells[1].startswith(u["gate"]) and cells[1].startswith(q["register_gate"]), (cells[1], nm["gate"])
        cite = "[" + "; ".join("@" + k for k in nm["cite"]) + "]" if nm["cite"] else "this project"
        out.append("| " + " | ".join([nm["use"], cite, u["needs"], BARFIRST.get(q["class"], "none"), plain(cells[5]),
                                      plain(cells[4]), plain(cells[6]), nm["code"]]) + " |")
    return "\n".join(out)


def read_bib():
    """refs.bib -> {key: (short label, reference line)} in file order."""
    bib = {}
    for m in re.finditer(r"@\w+\{(\w+),(.*?)\n\}", (HERE / "refs.bib").read_text(), re.S):
        f = dict(re.findall(r"(\w+)\s*=\s*\{(.*)\},?\s*$", m.group(2), re.M))
        clean = lambda s: re.sub(r"[{}]|\\url", "", s).replace("\\'e", "é")
        auth = [a.strip() for a in clean(f["author"]).split(" and ")]
        last = lambda a: a.split(",")[0]
        lab = last(auth[0]) + (" et al." if len(auth) > 2 else " and " + last(auth[1]) if len(auth) == 2 else "")
        where = clean(f.get("journal") or f.get("howpublished") or f.get("note", ""))
        bib[m.group(1)] = (lab, f"{', '.join(auth)}. {clean(f['title'])}. {where}, {f['year']}.")
    return bib


def numbering(src):
    """Tables and figures numbered by order of definition, citations by order of first use."""
    num, k = {}, {"tab": 0, "fig": 0}
    for m in re.finditer(r"^Table @(tab:\w+):|^!\[.*\]\(.*\)\{#(fig:\w+)", src, re.M):
        key = m.group(1) or m.group(2); k[key[:3]] += 1; num[key] = str(k[key[:3]])
    cites = []
    for m in re.finditer(r"\[(@[^\]]+)\]", src):
        for key in re.findall(r"@(\w+)", m.group(1)):
            if key not in cites: cites.append(key)
    return num, cites


def to_md(src, bib):
    num, cites = numbering(src)
    for c in cites:
        assert c in bib, f"citation {c} not in refs.bib"
    md = re.sub(r"\[(@[^\]]+)\]", lambda m: "[" + ", ".join(str(cites.index(k) + 1) for k in re.findall(r"@(\w+)", m.group(1))) + "]", src)
    md = re.sub(r"\{#fig:\w+\}", "", md)
    md = re.sub(r"@((?:tab|fig):\w+)", lambda m: num[m.group(1)], md)
    return md.rstrip("\n") + "\n\n## References\n\n" + "\n".join(f"[{i + 1}] {bib[k][1]}" for i, k in enumerate(cites)) + "\n"


SUB = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
       ("≥", r"$\geq$"), ("≈", r"$\approx$"), ("×", r"$\times$"), ("τ", r"$\tau$"), ("²", r"$^2$"), ("′", r"$'$"),
       ("—", "---"), ("–", "--"), ("…", r"\ldots{}")]


def esc(s):
    for a, b in SUB:
        s = s.replace(a, b)
    s = re.sub(r"`([^`]*)`", r"\\texttt{\1}", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    return re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)


COLS = {"tab:scale": (r"\scriptsize", r">{\raggedright}p{2.2cm}p{1.2cm}>{\raggedright}p{4.4cm}>{\raggedright}p{2.6cm}>{\raggedright}p{1.5cm}>{\raggedright}p{3.6cm}"), "tab:register": (r"\scriptsize", r">{\raggedright}p{2.2cm}>{\raggedright}p{1.0cm}p{1.25cm}>{\raggedright}p{0.9cm}>{\raggedright}p{2.1cm}>{\raggedright}p{4.2cm}>{\raggedright}p{1.5cm}p{1.1cm}")}


def refs(s):
    """esc'd text -> \\citep / \\ref (non-breaking space after Table / Fig. / Section)."""
    s = re.sub(r"\[(@[^\]]+)\]", lambda m: r"~\citep{" + ",".join(re.findall(r"@(\w+)", m.group(1))) + "}", s)
    s = s.replace(" ~\\citep", "~\\citep")
    return re.sub(r"(Table|Fig\.|Figure|Section) @((?:tab|fig|sec):\w+)", r"\1~\\ref{\2}", s)


def cell(s):
    return refs(esc(s)).lstrip("~")


LINKS = r"\textbf{Code and rows:} \texttt{github.com/ravediamond/paper-jepa-world-model}\quad\textbf{Reproduce:} \texttt{make check}. Every number carries its source file."


def tint(v):
    v = v.lower()
    c = "shissofail" if ("fail" in v or "no-go" in v) else "shissoamber" if "not proven" in v else "shissopass" if "pass" in v else None
    return "\\rowcolor{" + c + "}" if c else ""


def to_tex(md, bib):
    out = [r"\documentclass[10pt,twocolumn]{article}", r"\usepackage[margin=0.75in]{geometry}", r"\usepackage{booktabs}", r"\usepackage{array}", r"\usepackage{graphicx}",
           r"\usepackage[numbers,sort&compress]{natbib}", r"\usepackage{style/shisso}", r"\usepackage{listings}\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,frame=tb,framerule=0.3pt,aboveskip=4pt,belowskip=4pt,upquote=true}", r"\shissorepo{github.com/ravediamond/paper-jepa-world-model}",
           r"\renewcommand{\dbltopfraction}{0.95}\renewcommand{\topfraction}{0.9}\renewcommand{\textfraction}{0.05}\setcounter{dbltopnumber}{2}", r"\begin{document}"]
    lines, i, abstract, cap, lab = md.splitlines(), 0, False, "", ""
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):  # fenced code from the repository: printed verbatim
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("\\begin{lstlisting}\n" + "\n".join(lines[i + 1:j]) + "\n\\end{lstlisting}")
            i = j + 1
            continue
        if l.startswith("# "):
            title = esc(l[2:]); out += [r"\title{" + title + "}",
                    r"\author{Ravindu Somawansa\thanks{Code, data pointers and every number's source: \url{https://github.com/ravediamond/paper-jepa-world-model}.}\\Shisso Robotics, Paris\\\texttt{ravindu.somawansa@shissorobotics.com}}",
                    r"\date{}"]
        elif l.startswith("## Abstract"):
            abs_at = len(out); abstract = True
        elif l.startswith("## "):
            if abstract:
                body = " ".join(x.strip() for x in out[abs_at:] if x.strip()); del out[abs_at:]; abstract = False
                out.append(r"\twocolumn[\vspace*{0.32in}\shissopanel{" + title + "}{Ravindu Somawansa\\\\[1pt]Shisso Robotics, Paris\\\\[1pt]\\texttt{ravindu.somawansa@shissorobotics.com}}{" + body + "}{" + LINKS + r"}\vspace{4pt}]")
            out.append(r"\section{" + esc(l[3:]) + "}")
        elif l.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)\{#(fig:\w+)\}(\{wide\}|\{full\})?", l); env = "figure*" if m.group(4) else "figure"
            out += [r"\begin{" + env + "}[t]\\centering", r"\includegraphics[width=" + (r"\textwidth" if m.group(4) == "{full}" else r"0.72\textwidth" if m.group(4) else r"\columnwidth") + "]{" + m.group(2) + "}",
                    r"\caption{" + cell(m.group(1)) + r"}\label{" + m.group(3) + "}", r"\end{" + env + "}"]
        elif re.match(r"Table @tab:\w+: ", l):
            lab, cap = re.match(r"Table @(tab:\w+): (.*)", l).groups()
        elif l.startswith("|"):
            t = []
            while i < len(lines) and lines[i].startswith("|"):
                if not set(lines[i]) <= set("|-: "):
                    t.append([cell(x.strip()) for x in lines[i].strip("|").split("|")])
                i += 1
            size, spec = COLS.get(lab, (r"\scriptsize", "l" * len(t[0])))
            end, env = (r" \tabularnewline", "table*") if lab in COLS else (r" \\", "table")
            out += [r"\begin{" + env + r"}[!t]\centering\setlength{\tabcolsep}{3pt}" + size, r"\caption{" + cell(cap) + r"}\label{" + lab + "}", (r"\resizebox{\columnwidth}{!}{" if env == "table" else "") + r"\begin{tabular}{" + spec + "}", r"\toprule",
                    r"\shissohead " + " & ".join(t[0]) + end, r"\midrule"] + [(tint(r[6]) if lab == "tab:register" else "") + " & ".join(r) + end for r in t[1:]] + \
                   [r"\bottomrule", r"\end{tabular}" + ("}" if env == "table" else ""), r"\end{" + env + "}"]
            continue
        elif l.startswith("*Ravindu"):
            pass
        elif l.startswith("> "):
            mm = re.match(r"\*\*(.+?)\*\*\s*(.*)", l[2:]); i += 1
            out.append(r"\begin{shissofinding}[" + cell(mm.group(1)) + "]\n" + cell(mm.group(2)) + "\n" + r"\end{shissofinding}")
            continue
        elif not l.strip():
            out.append("\n")  # keep paragraph breaks: an empty markdown line ends the paragraph in LaTeX too
        else:
            out.append(cell(l))
        i += 1
    if abstract:
        out.append(r"\end{abstract}")
    out += [r"\bibliographystyle{unsrtnat}", r"\bibliography{refs}", r"\end{document}"]
    return "\n".join(x for x in out if x != "") + "\n"


if __name__ == "__main__":
    main()
