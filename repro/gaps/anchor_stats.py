"""Table 2 grasp-anchor column from expected/anchor_predictions.csv (standard library only).

Statistic (the table's): per feature, the mean over the 5 seeds of the out-of-fold AUROC (label 1 = failure).
CI: 95 % episode bootstrap of that statistic: 2,000 resamples, random.Random(0); each resample draws episodes with
replacement once, jointly for all seeds and features; resamples with one class are skipped; percentile interval.

    python3 anchor_stats.py            # from repro/gaps: writes expected/anchor_ci_mean_of_seeds.json
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

E = Path(__file__).resolve().parent / "expected"
FEATURES = ("W1", "W2p", "joints")
RESAMPLES, SEED = 2000, 0


def auroc(y, s):
    """Mann-Whitney AUROC, ties counted 1/2."""
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    w = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def load(path=E / "anchor_predictions.csv"):
    """-> episodes (sorted), labels {episode: label}, scores {feature: {seed: {episode: score}}}."""
    lab, sc = {}, defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            e = int(r["episode"])
            lab[e] = int(r["label"])
            sc[r["feature"]][int(r["seed"])][e] = float(r["score"])
    return sorted(lab), lab, sc


def mean_of_seeds(idx, eps, lab, by_seed):
    y = [lab[eps[i]] for i in idx]
    per = [auroc(y, [by_seed[k][eps[i]] for i in idx]) for k in sorted(by_seed)]
    return sum(per) / len(per), per


def pct(v, q):
    """numpy's default (linear) percentile."""
    v = sorted(v); h = (len(v) - 1) * q / 100; lo = int(h)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (h - lo)


def compute():
    eps, lab, sc = load()
    n, full = len(eps), list(range(len(eps)))
    rng, boots, used = random.Random(SEED), {k: [] for k in FEATURES}, 0
    for _ in range(RESAMPLES):
        idx = [rng.randrange(n) for _ in range(n)]
        if not 0 < sum(lab[eps[i]] for i in idx) < n:
            continue
        used += 1
        for k in FEATURES:
            boots[k].append(mean_of_seeds(idx, eps, lab, sc[k])[0])
    out = {"source": "expected/anchor_predictions.csv", "statistic": "mean over 5 seeds of the out-of-fold AUROC",
           "bootstrap": {"resamples": RESAMPLES, "used": used, "rng": f"random.Random({SEED})", "unit": "episode, joint across seeds",
                         "interval": "percentile 95 %"},
           "scored": n, "failures": sum(lab.values()), "features": {}}
    for k in FEATURES:
        m, per = mean_of_seeds(full, eps, lab, sc[k])
        avg = [sum(sc[k][s][e] for s in sc[k]) / len(sc[k]) for e in eps]
        out["features"][k] = {"auroc_mean_of_seeds": round(m, 4), "auroc_oof_per_seed": [round(p, 4) for p in per],
                              "ci95_mean_of_seeds": [round(pct(boots[k], 2.5), 4), round(pct(boots[k], 97.5), 4)],
                              "auroc_seed_averaged": round(auroc([lab[e] for e in eps], avg), 4),
                              "auroc_seed0": round(per[0], 4)}
    return out


if __name__ == "__main__":
    r = compute()
    (E / "anchor_ci_mean_of_seeds.json").write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r["features"]))
