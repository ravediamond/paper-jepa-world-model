"""WS-106 probe battery (PROTOCOL.md §3; FREEZE_2026-11-02.md). One checkpoint per call; the original scripts do the
work, this file adds only what the protocol asks beyond them (paired floors, the verdict per probe, check C2).

  P1  gate0_extract.py + gate0_analyze.py (projects/lejepa-identifiability), --w1 = this checkpoint      [run_battery.sh]
  P2  p2:     g1b_range (W1 candidate = this checkpoint) + the action-only ranker of g1_diag, paired by episode
  P3  p3lat:  id_latents.encode_set with W1 = this checkpoint; W1_random_init / W2prime latents reused from the §30 run
              (the same untrained encoder at every scale, C7); then id_probe.py unchanged (ridge)               [run_battery.sh]
  P4  run_w1_offline_screening.py --w1 this checkpoint (W1-A part)                                          [run_battery.sh]
  P5  p5:     analyze_controls.loso / boot on P4's cached latents at the judge frame; floor = G9c-3 judge score (O1), paired
  verdict:    one JSON per scale; c2: W1's verdict against the published values (PROTOCOL.md §6 C2)

  WM_SPLIT=<wm_train/split.json> python probes.py p2 <weights dir> --data <so101-bowls-h264-224> --out <p2.json>
  WM_SPLIT=... python probes.py p3lat <weights dir> --teleop <h264-224> --rollouts <root> --outcomes <csv> --ref <id30 dir> --out <dir>
  python3 probes.py p5 <latents dir> --out <p5.json>
  python3 probes.py verdict <scale results dir>      # writes <dir>/verdict.json
  python3 probes.py c2 <W1 results dir>              # exit 1 on a mismatch
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WT = ROOT / "projects/vla-jepa-world-model/wm_train"
GAPS = ROOT / "journal/papers/jepa-world-model/gaps"
N_BOOT = 2000


def paired_boot(eps, a, b, n=N_BOOT):
    """Episode bootstrap of mean(a) - mean(b) over anchors (a, b per anchor), the same resampled episodes for both."""
    uniq = np.unique(eps)
    per = np.stack([[a[eps == u].sum(), b[eps == u].sum(), (eps == u).sum()] for u in uniq])
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        x = per[rng.integers(len(uniq), size=len(uniq))].sum(0)
        out.append((x[0] - x[1]) / x[2])
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def cmd_p2(a):
    import torch
    sys.path.insert(0, str(WT))
    import g1b_range as G
    import g1_diag as D
    import probe as P

    torch.manual_seed(0)
    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    eps = split["val_episodes"]
    cand = G.W1(a.weights)
    a_mean, a_std = P.action_stats(a.data, split["train_episodes"])
    tab = P.frame_table(a.data)
    a_norm = {ep: ((tab[ep][1] - a_mean) / a_std).astype(np.float32) for ep in eps}
    s_norm = {ep: ((tab[ep][2] - a_mean) / a_std).astype(np.float32) for ep in eps}
    Z = G.embed_episodes(cand, a.data, tab, eps, a.workers)
    anchors, chunks = G.build_chunks(a_norm, eps, np.random.default_rng(0))
    dist, _, _ = G.rollout_stream(cand, Z, anchors, chunks)
    cost = D.end_state_dist(chunks, anchors, s_norm)
    ep_arr = np.array([e for e, _ in anchors])

    def hits(s):
        s = s[:, :8]
        return ((1 + (s[:, 1:] <= s[:, :1]).sum(1)) == 1).numpy().astype(float)  # ties count against the true chunk

    model = G.cells_from(dist, anchors)
    floor = G.cells_from({(d, "d"): cost[d] for d in G.DS}, anchors)
    per_d, arr = {}, {"episode": ep_arr}
    for d in G.DS:
        hm, hf = hits(dist[(d, "d")]), hits(cost[d])
        arr[f"model_d{d}"], arr[f"floor_d{d}"] = hm, hf
        per_d[f"d{d}"] = {"model_top1": model[f"d{d}_Hd"]["accuracy"], "model_ci95": model[f"d{d}_Hd"]["ci95"],
                          "action_only_top1": floor[f"d{d}_Hd"]["accuracy"], "action_only_ci95": floor[f"d{d}_Hd"]["ci95"],
                          "diff": round(float(hm.mean() - hf.mean()), 4), "diff_ci95": paired_boot(ep_arr, hm, hf)}
    res = {"weights": str(a.weights), "anchors": len(anchors), "val_episodes": len(eps), "H": "d", "cells_H_d": per_d,
           "model_all_cells": model, "protocol": "PROTOCOL.md §3 P2 / P2-floor; g1b_range + g1_diag.end_state_dist"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    np.savez_compressed(a.out.with_suffix(".npz"), **arr)
    print(json.dumps(per_d), flush=True)


def cmd_p3lat(a):
    import csv
    sys.path.insert(0, str(WT))
    import id_latents as L
    from action_sensitivity import W1Like

    split = json.loads(Path(os.environ["WM_SPLIT"]).read_text())
    enc = {"W1": W1Like(a.weights, False)}
    rows = list(csv.DictReader(open(a.outcomes)))
    sets = {"teleop_train": (a.teleop, split["train_episodes"], False),
            "teleop_val": (a.teleop, split["val_episodes"], False),
            "rollouts": (a.rollouts, sorted(int(r["episode_index"]) for r in rows), True)}
    for name, (root, eps, resize) in sets.items():
        d = a.out / name
        if not (d / "W1.npz").exists():
            L.encode_set(root, sorted(eps), resize, enc, d)
        for f in ("W1_random_init.npz", "W2prime.npz"):   # scale-independent: the §30 run's own files (C7)
            if not (d / f).exists():
                (d / f).symlink_to(a.ref / name / f)


def cmd_p3probe(a):
    """id_probe.py unchanged, without the W2prime cells (scale-independent, computed once at S3); W1_random_init stays,
    the P3 rule reads it."""
    sys.path.insert(0, str(WT))
    import id_probe as I

    I.ENCODERS = ("W1", "W1_random_init")
    sys.argv = ["id_probe.py", "--latents", str(a.latents), "--out", str(a.out)]
    I.main()


def cmd_p5(a):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(GAPS))
    import analyze_controls as AC

    facts = json.loads((ROOT / "data/FACTS.json").read_text())["rollouts"]
    root = (Path.home() / ".cache/huggingface/hub" / ("datasets--" + facts["repo"].replace("/", "--"))
            / "snapshots" / facts["revision"])
    if not root.exists():   # the local download of the same revision (its huggingface metadata names the commit)
        root = Path.home() / "robot-data/so101-bowls-policy-rollouts"
        meta = root / ".cache/huggingface/download/meta/rollout_labels.csv.metadata"
        assert meta.read_text().split()[0] == facts["revision"], meta
    lab = pd.read_csv(root / "meta/rollout_labels.csv")
    lab = lab[lab.outcome.isin(["success", "failure"])].reset_index(drop=True)
    judge = pd.read_csv(ROOT / "projects/surveyor/results/judge.csv").set_index("episode")
    eps = lab.episode_index.astype(int).to_numpy()
    y = (lab.outcome == "success").astype(int).to_numpy()
    groups = lab.session.astype(str).to_numpy()
    X = np.stack([np.load(a.latents / f"ep{e:03d}.npy")[int(judge.loc[e, "judge_frame"])] for e in eps]).astype(np.float64)
    assert np.isfinite(X).all()
    s = AC.loso(X, y, groups)
    j = np.array([float(judge.loc[e, "s"]) for e in eps])
    ok = np.array([not bool(judge.loc[e, "invalid"]) for e in eps])
    r, diffs = np.random.default_rng(0), []          # paired: the same resampled valid episodes for model and judge
    yo, so, jo = y[ok], s[ok], j[ok]
    for _ in range(N_BOOT):
        i = r.integers(0, len(yo), len(yo))
        if 0 < yo[i].sum() < len(i):
            diffs.append(roc_auc_score(yo[i], so[i]) - roc_auc_score(yo[i], jo[i]))
    res = {"latents": str(a.latents), "n": len(y), "success": int(y.sum()), "sessions": len(set(groups)),
           "auroc": round(float(roc_auc_score(y, s)), 4), "ci95": AC.boot(y, s),
           "judge_auroc_valid": round(float(roc_auc_score(yo, jo)), 4), "n_valid": int(ok.sum()),
           "model_auroc_valid": round(float(roc_auc_score(yo, so)), 4),
           "diff_vs_judge_valid": round(float(roc_auc_score(yo, so) - roc_auc_score(yo, jo)), 4),
           "diff_ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
           "protocol": "PROTOCOL.md §3 P5; floor = G9c-3 judge score (AMENDMENT O1), paired on the valid-judge episodes"}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


def verdict(d: Path) -> dict:
    v = {"dir": str(d)}
    g = d / "p1/gate0.json"
    if g.exists():
        m = json.loads(g.read_text())["models"]["W1"]
        v["P1"] = {k: m.get(k) for k in ("mean_r2", "anisotropy")} | {"source": str(g)}
    g = d / "p2.json"
    if g.exists():
        c = json.loads(g.read_text())["cells_H_d"]
        ds = [int(k[1:]) for k in c]
        bar = [x for x in ds if c[f"d{x}"]["model_top1"] >= 0.8]
        flo = [x for x in ds if c[f"d{x}"]["diff_ci95"][0] > 0 and c[f"d{x}"]["diff"] > 0]
        v["P2"] = {"best_top1": max(c[k]["model_top1"] for k in c), "largest_d_bar_0.8": max(bar, default=None),
                   "largest_d_floor": max(flo, default=None), "largest_d_bar_and_floor": max(set(bar) & set(flo), default=None),
                   "cells": c}
    g = d / "p3/results_ridge.json"
    if g.exists():
        r = json.loads(g.read_text())
        v["P3"] = {"bar": r["bar"], "diffs_W1_both_5": r["cells"]["W1|both|5|ridge"]["diffs"]["teleop_val"]}
    g = d / "p4/w1a_summary.json"
    if g.exists():
        r = json.loads(g.read_text())
        v["P4"] = {s: {k: r[s][k] for k in ("l2", "cos")} for s in ("g16_set", "all_69") if s in r}
        v["P4"]["verdict_source"] = r.get("verdict")
    g = d / "p5.json"
    if g.exists():
        v["P5"] = json.loads(g.read_text())
    (d / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    return v


def cmd_c2(a):
    v = verdict(a.dir)
    pub = {"P1 mean R2": (v["P1"]["mean_r2"], 0.5654), "P1 anisotropy": (v["P1"]["anisotropy"], 0.6393),
           "P2 best cell": (v["P2"]["best_top1"], 0.6046),
           "P3 arm (ii)-max": (v["P3"]["bar"]["arm_ii_minus_max_i_iii"], 0.748),
           "P3 arm (v)-(iv)": (v["P3"]["bar"]["arm_v_minus_iv"]["value"], 0.007),
           "P4 caught (L2, 50)": (v["P4"]["g16_set"]["l2"]["caught_failed"], 2),
           "P4 FI rate (L2, 50)": (v["P4"]["g16_set"]["l2"]["false_intervention_rate"], round(2 / 42, 4)),
           "P5 AUROC": (v["P5"]["auroc"], 0.9545)}
    rows, bad = [], 0
    for k, (got, ref) in pub.items():
        dec = len(str(ref).split(".")[1]) if "." in str(ref) else 0
        ok = round(float(got), dec) == ref
        bad += not ok
        rows.append({"check": k, "probes.py": got, "published": ref, "match": ok})
    (a.dir / "c2.json").write_text(json.dumps(rows, indent=1, default=float))
    for r in rows:
        print(f"{r['check']:22s} {r['probes.py']!s:>10} {r['published']!s:>8} {'ok' if r['match'] else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


SCALES = {"S1": "splits/S1/split.json", "S3": None, "S4": "splits/S4/split.json", "S5a": "splits/S5a/split.json"}


def train_frames(scale: str, probes: Path) -> int:
    """Training frames (PROTOCOL.md §2 x-axis): bowls train episodes from the §30 frame table (every frame encoded),
    plus each extra dataset's frames from c3_meta.json (all their episodes train)."""
    here = Path(__file__).resolve().parent
    m = np.load(probes / "S3/p3lat/teleop_train/meta.npz")
    length = {int(k[2:5]): len(m[k]) for k in m.files if k.endswith("_frame")}
    sp = json.loads((here / SCALES[scale]).read_text()) if SCALES[scale] else {"train_episodes": sorted(length)}
    n = sum(length[e] for e in sp["train_episodes"])
    c3 = json.loads((here / "c3_meta.json").read_text())
    for x in sp.get("extra_train", []) or []:
        if isinstance(x, dict) and x.get("frames"):  # the split's own count (S5a trains on the 2,099 rollouts, not all 2,499 episodes)
            n += x["frames"]; continue
        name = x["repo_id"].split("/")[1] if isinstance(x, dict) else str(x).split("/")[1]
        n += next(v["frames"] for v in c3.values() if isinstance(v, dict) and v.get("repo", "").endswith(name))
    return n


def cmd_table(a):
    rows, out = [], {}
    for s in SCALES:
        f = a.probes / s / "verdict.json"
        if not f.exists():
            continue
        v = json.loads(f.read_text())
        out[s] = v | {"train_frames": train_frames(s, a.probes)}
        p2, p3, p4, p5 = v.get("P2", {}), v.get("P3", {}).get("bar", {}), v.get("P4", {}).get("g16_set", {}), v.get("P5", {})
        c30 = p2.get("cells", {}).get("d30", {})
        rows.append(f"| {s} | {out[s]['train_frames']} | {v.get('P1', {}).get('mean_r2')} / {v.get('P1', {}).get('anisotropy')} "
                    f"| {p2.get('best_top1')} | {c30.get('diff')} {c30.get('diff_ci95')} | {p2.get('largest_d_floor')} "
                    f"| {round(p3['arm_ii_minus_max_i_iii'], 3) if p3 else None} | {round(p3['arm_v_minus_iv']['value'], 4) if p3 else None} "
                    f"| {p3.get('branch')} | {p4.get('l2', {}).get('caught_failed')}/8 at {p4.get('l2', {}).get('false_intervention_rate')} "
                    f"| {p5.get('auroc')} {p5.get('ci95')} | {p5.get('diff_vs_judge_valid')} {p5.get('diff_ci95')} |")
    head = ("| scale | train frames | P1 R² / anisotropy | P2 best top-1 | P2 d30 − action-only [CI] | P2 largest d over floor "
            "| P3 arm (ii)−max | P3 arm (v)−(iv) | P3 branch | P4 L2 caught at FI | P5 AUROC [CI] | P5 − judge [CI] |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "curve.json").write_text(json.dumps(out, indent=1, default=float))
    (a.out / "curve.md").write_text(head + "\n" + "\n".join(rows) + "\n")
    print(head + "\n" + "\n".join(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("p2"); p.add_argument("weights", type=Path); p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--workers", type=int, default=4)
    p = sp.add_parser("p3lat"); p.add_argument("weights", type=Path)
    for k in ("teleop", "rollouts", "outcomes", "ref", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p = sp.add_parser("p3probe"); p.add_argument("latents", type=Path); p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("p5"); p.add_argument("latents", type=Path); p.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("table"); p.add_argument("probes", type=Path); p.add_argument("--out", type=Path, required=True)
    for k in ("verdict", "c2"):
        p = sp.add_parser(k); p.add_argument("dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "verdict":
        print(json.dumps(verdict(a.dir), default=float)[:2000])
    else:
        {"p2": cmd_p2, "p3lat": cmd_p3lat, "p3probe": cmd_p3probe, "p5": cmd_p5, "c2": cmd_c2, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
