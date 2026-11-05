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
    import make_figs; make_figs.OUT.mkdir(exist_ok=True); make_figs.table2()
    print(f"{len(N)} numbers; DRAFT_4p.md, main.tex, numbers.json written")


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


COLS = {"tab:register": (r"\scriptsize", r">{\raggedright}p{2.2cm}>{\raggedright}p{1.0cm}p{1.25cm}>{\raggedright}p{0.9cm}>{\raggedright}p{2.1cm}>{\raggedright}p{4.2cm}>{\raggedright}p{1.5cm}p{1.1cm}")}


def refs(s):
    """esc'd text -> \\citep / \\ref (non-breaking space after Table / Fig. / Section)."""
    s = re.sub(r"\[(@[^\]]+)\]", lambda m: r"~\citep{" + ",".join(re.findall(r"@(\w+)", m.group(1))) + "}", s)
    s = s.replace(" ~\\citep", "~\\citep")
    return re.sub(r"(Table|Fig\.|Figure|Section) @((?:tab|fig|sec):\w+)", r"\1~\\ref{\2}", s)


def cell(s):
    return refs(esc(s)).lstrip("~")


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
            out += [r"\title{" + esc(l[2:]) + "}",
                    r"\author{Ravindu Somawansa\thanks{Code, data pointers and every number's source: \url{https://github.com/ravediamond/paper-jepa-world-model}.}\\Shisso Robotics, Paris\\\texttt{ravindu.somawansa@shissorobotics.com}}",
                    r"\date{}", r"\maketitle"]
        elif l.startswith("## Abstract"):
            out.append(r"\begin{abstract}"); abstract = True
        elif l.startswith("## "):
            if abstract:
                out.append(r"\end{abstract}"); abstract = False
            out.append(r"\section{" + esc(l[3:]) + "}")
        elif l.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)\{#(fig:\w+)\}(\{wide\})?", l); env = "figure*" if m.group(4) else "figure"
            out += [r"\begin{" + env + "}[t]\\centering", r"\includegraphics[width=" + (r"0.72\textwidth" if m.group(4) else r"\columnwidth") + "]{" + m.group(2) + "}",
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
                    r"\shissohead " + " & ".join(t[0]) + end, r"\midrule"] + [" & ".join(r) + end for r in t[1:]] + \
                   [r"\bottomrule", r"\end{tabular}" + ("}" if env == "table" else ""), r"\end{" + env + "}"]
            continue
        elif l.startswith("*Ravindu"):
            pass
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
