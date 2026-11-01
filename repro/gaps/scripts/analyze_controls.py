"""WS-091 gaps 2-3 (exploratory, in-sample): rest readout and W1-C1 / SURVEYOR controls with W2' and commanded joints.

    python3 analyze_controls.py <controls dir from extract_controls.py> <out dir>

Labels: meta/rollout_labels.csv of the rollouts @ data/FACTS.json revision (outcome; policy; session).
Gap 2: G10 step 1 readout at the judge frame (projects/surveyor/results/judge.csv, G9c-3 rerun at that revision):
leave-one-session-out L2 logistic (StandardScaler, C = 1.0, balanced), as wm_train/g10_eval.fit_loso, for W1
(outputs/reap/latents), W2', commanded joints, policy identity (one-hot policy); the G9c-3 score s with no learning.
Gap 3: W1-C1's probe_cv (scripts/run_w1c_offline.py) on the current feature at its anchor; SURVEYOR gate 0
(projects/surveyor/scripts/surveyor_gate0.py) run on the W2' and joints directories.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from run_w1c_offline import K_IMAGINE, c1_anchor, probe_cv  # noqa: E402


def loso(X, y, groups):
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        sc = StandardScaler().fit(X[~te])
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X[~te]), y[~te])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def boot(y, s, n=2000, seed=0):
    r, out = np.random.default_rng(seed), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def main():
    ctl, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    facts = json.loads((ROOT / "data/FACTS.json").read_text())["rollouts"]
    root = (Path.home() / ".cache/huggingface/hub" / ("datasets--" + facts["repo"].replace("/", "--"))
            / "snapshots" / facts["revision"])
    lab = pd.read_csv(root / "meta/rollout_labels.csv")
    lab = lab[lab.outcome.isin(["success", "failure"])].reset_index(drop=True)
    judge = pd.read_csv(ROOT / "projects/surveyor/results/judge.csv").set_index("episode")
    eps = lab.episode_index.astype(int).to_numpy()
    feats = {"W1": ROOT / "outputs/reap/latents", "W2'": ctl / "w2p", "joints (commanded)": ctl / "joints"}
    Z = {k: {e: np.load(d / f"ep{e:03d}.npy") for e in eps} for k, d in feats.items()}
    y = (lab.outcome == "success").astype(int).to_numpy()
    groups = lab.session.astype(str).to_numpy()
    res = {"n": len(y), "success": int(y.sum()), "failure": int((1 - y).sum()), "sessions": len(set(groups)),
           "labels": f"{facts['repo']}@{facts['revision']} meta/rollout_labels.csv", "in_sample": True, "gap2": {}}
    jf = {e: int(judge.loc[e, "judge_frame"]) for e in eps}
    for k in feats:
        X = np.stack([Z[k][e][jf[e]] for e in eps]).astype(np.float64)
        assert np.isfinite(X).all(), k
        s = loso(X, y, groups)
        res["gap2"][f"{k} at judge frame (LOSO probe)"] = {"auroc": round(roc_auc_score(y, s), 4), "ci95": boot(y, s)}
    pol = pd.get_dummies(lab.policy.astype(str)).to_numpy(float)
    s = loso(pol, y, groups)
    res["gap2"]["policy identity (LOSO probe)"] = {"auroc": round(roc_auc_score(y, s), 4), "ci95": boot(y, s)}
    s = np.array([float(judge.loc[e, "s"]) for e in eps])
    ok = np.array([not bool(judge.loc[e, "invalid"]) for e in eps])
    res["gap2"]["G9c-3 judge score s (no learning)"] = {"auroc": round(roc_auc_score(y[ok], s[ok]), 4),
                                                        "ci95": boot(y[ok], s[ok]), "n_valid": int(ok.sum())}
    # gap 3: W1-C1 current feature at the anchor (same rule and exclusions as run_w1c_offline)
    rows = []
    for r in lab.itertuples():
        e = int(r.episode_index)
        acts = Z["joints (commanded)"][e]
        T = len(acts)
        end = T if int(r.reset_start) < 0 else min(int(r.reset_start), T)
        a = c1_anchor(acts[:end, 5])
        if a is None or a < 2 or a + K_IMAGINE > T or np.isnan(Z["W1"][e][a - 2: a + 1]).any():
            continue
        rows.append((e, a, int(r.outcome == "failure")))
    yc = np.array([x[2] for x in rows])
    res["w1c1"] = {"scored": len(rows), "failures": int(yc.sum())}
    for k in feats:
        X = np.stack([Z[k][e][a] for e, a, _ in rows]).astype(np.float64)
        cv = probe_cv(X, yc)
        res["w1c1"][f"{k} current at anchor"] = {kk: cv[kk] for kk in cv if "auroc" in kk.lower()}
    # gap 3: SURVEYOR gate 0 on each feature directory
    res["surveyor_gate0"] = {}
    for k, d in feats.items():
        o = out / ("surveyor_" + k.split()[0].replace("'", "p"))
        o.mkdir(exist_ok=True)
        shutil.copy(ROOT / "projects/surveyor/results/judge.csv", o / "judge.csv")
        subprocess.run([sys.executable, str(ROOT / "projects/surveyor/scripts/surveyor_gate0.py"), str(d), str(o)],
                       check=True, capture_output=True)
        g = json.loads((o / "gate0.json").read_text())
        res["surveyor_gate0"][k] = {"tau": g["bars"]["S2"]["tau"], "ne_p10": g["bars"]["S1"]["ne_p10"],
                                    "auroc_eq_vs_ne": g["auroc_eq_vs_ne"], "S1_met": g["bars"]["S1"]["met"],
                                    "S2_met": g["bars"]["S2"]["met"]}
    (out / "controls.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
