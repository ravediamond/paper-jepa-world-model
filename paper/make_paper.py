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
    put("bar_first", str(pre["bar-first"]), a + " (pre-reg. bar-first)"); put("same_commit", str(pre["same-commit"]), a + " (pre-reg. same-commit)")
    put("no_class", str(pre["—"]), a + " (pre-reg. '—')")
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

    tpl = (HERE / "DRAFT_4p.md.in").read_text()
    md = re.sub(r"\{\{(\w+)\}\}", lambda m: N[m.group(1)], tpl)
    assert "{{" not in md
    (HERE / "DRAFT_4p.md").write_text(md)
    (HERE / "numbers.json").write_text(json.dumps({k: {"value": N[k], "anchor": A[k]} for k in sorted(N)}, indent=1, ensure_ascii=False) + "\n")
    (HERE / "main.tex").write_text(to_tex(md))
    import make_figs; make_figs.OUT.mkdir(exist_ok=True); make_figs.table2(); make_figs.register()
    print(f"{len(N)} numbers; DRAFT_4p.md, main.tex, numbers.json written")


SUB = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
       ("≥", r"$\geq$"), ("≈", r"$\approx$"), ("×", r"$\times$"), ("τ", r"$\tau$"), ("²", r"$^2$"), ("′", r"$'$"),
       ("—", "---"), ("–", "--"), ("…", r"\ldots{}")]


def esc(s):
    for a, b in SUB:
        s = s.replace(a, b)
    s = re.sub(r"`([^`]*)`", r"\\texttt{\1}", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    return re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)


def to_tex(md):
    out = [r"\documentclass[10pt,twocolumn]{article}", r"\usepackage[margin=0.75in]{geometry}", r"\usepackage{booktabs}", r"\usepackage{graphicx}", r"\usepackage{style/shisso}", r"\shissorepo{github.com/ravediamond/paper-jepa-world-model}",
           r"\pdfinfoomitdate=1 \pdftrailerid{}" if False else "", r"\begin{document}"]
    lines, i, abstract, cap = md.splitlines(), 0, False, ""
    while i < len(lines):
        l = lines[i]
        if l.startswith("# "):
            out += [r"\title{" + esc(l[2:]) + "}", r"\author{Ravindu Somawansa\\Shisso Robotics\\\texttt{ravindu.somawansa@gmail.com}}", r"\date{}", r"\maketitle"]
        elif l.startswith("## Abstract"):
            out.append(r"\begin{abstract}"); abstract = True
        elif l.startswith("## "):
            if abstract:
                out.append(r"\end{abstract}"); abstract = False
            out.append(r"\section*{" + esc(l[3:]) + "}")
        elif l.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)(\{wide\})?", l); env = "figure*" if m.group(3) else "figure"
            out += [r"\begin{" + env + "}[t]\\centering", r"\includegraphics[width=" + (r"0.72\textwidth" if m.group(3) else r"\columnwidth") + "]{" + m.group(2) + "}",
                    r"\caption{" + esc(m.group(1)) + "}", r"\end{" + env + "}"]
        elif l.startswith("Table ") and i + 2 < len(lines) and lines[i + 2].startswith("|"):
            cap = l.split(": ", 1)[1]
        elif l.startswith("|"):
            t = []
            while i < len(lines) and lines[i].startswith("|"):
                if not set(lines[i]) <= set("|-: "):
                    t.append([esc(x.strip()) for x in lines[i].strip("|").split("|")])
                i += 1
            out += [r"\begin{table*}[t]\centering\footnotesize", r"\begin{tabular}{" + "l" * len(t[0]) + "}", r"\toprule",
                    r"\shissohead " + " & ".join(t[0]) + r" \\", r"\midrule"] + [" & ".join(r) + r" \\" for r in t[1:]] + \
                   [r"\bottomrule", r"\end{tabular}", r"\caption{" + esc(cap) + "}", r"\end{table*}"]
            continue
        elif l.startswith("*Ravindu"):
            pass
        else:
            out.append(esc(l))
        i += 1
    if abstract:
        out.append(r"\end{abstract}")
    out.append(r"\end{document}")
    return "\n".join(x for x in out if x != "") + "\n"


if __name__ == "__main__":
    main()
