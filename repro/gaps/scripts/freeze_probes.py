#!/usr/bin/env python3
"""PROTOCOL_CONFIRM.md §3: fit the rest probes once on all 69 old rollouts and freeze them (no refit on fresh rollouts).

usage: freeze_probes.py <road-to-robotics root> <controls dir (w2p/, joints/)> <out dir>
Writes <out>/frozen_probes.npz (per feature: scaler mean / scale, coef, intercept; policy one-hot columns) and
<out>/frozen_probes.json (sha256 of the npz, fit n, the in-sample AUROC sanity of §6.3). Estimator and frame as
analyze_controls.py (judge_frame, StandardScaler + LogisticRegression(C=1.0, balanced, max_iter=5000)).
The grasp-anchor probes (current / imagined) need W1's imagination in the strict-load image; they are added by the
same script's anchor step before the first fresh rollout (PROTOCOL_CONFIRM.md §3)."""
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def main():
    root, ctl, out = (Path(a) for a in sys.argv[1:4])
    facts = json.loads((root / "data/FACTS.json").read_text())["rollouts"]
    snap = (Path.home() / ".cache/huggingface/hub" / ("datasets--" + facts["repo"].replace("/", "--"))
            / "snapshots" / facts["revision"])
    lab = pd.read_csv(snap / "meta/rollout_labels.csv")
    lab = lab[lab.outcome.isin(["success", "failure"])].reset_index(drop=True)
    judge = pd.read_csv(root / "projects/surveyor/results/judge.csv").set_index("episode")
    eps = lab.episode_index.astype(int).to_numpy()
    y = (lab.outcome == "success").astype(int).to_numpy()
    jf = {e: int(judge.loc[e, "judge_frame"]) for e in eps}
    feats = {"W1": root / "outputs/reap/latents", "W2p": ctl / "w2p", "joints": ctl / "joints"}
    X = {k: np.stack([np.load(d / f"ep{e:03d}.npy")[jf[e]] for e in eps]).astype(np.float64) for k, d in feats.items()}
    pol = pd.get_dummies(lab.policy.astype(str))
    X["policy"] = pol.to_numpy(float)
    arrays, meta = {"policy_columns": np.array(pol.columns, dtype=str)}, {"n": int(len(y)), "success": int(y.sum()),
                                                                           "labels": f"{facts['repo']}@{facts['revision']}", "features": {}}
    for k, x in X.items():
        assert np.isfinite(x).all(), k
        sc = StandardScaler().fit(x)
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(x), y)
        arrays.update({f"{k}_mean": sc.mean_, f"{k}_scale": sc.scale_, f"{k}_coef": m.coef_[0], f"{k}_intercept": m.intercept_})
        meta["features"][k] = {"dim": int(x.shape[1]), "in_sample_auroc_sanity": round(float(roc_auc_score(y, m.decision_function(sc.transform(x)))), 4)}
    out.mkdir(parents=True, exist_ok=True)
    f = out / "frozen_probes.npz"
    np.savez(f, **arrays)
    meta["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    meta["anchor_probes"] = "pending: strict-load image step, before the first fresh rollout"
    (out / "frozen_probes.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v["in_sample_auroc_sanity"] for k, v in meta["features"].items()}), meta["sha256"][:12])


if __name__ == "__main__":
    main()
