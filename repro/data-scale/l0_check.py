#!/usr/bin/env python3
"""repro/data-scale L0: recompute every per-scale verdict from the stored probe outputs and check the statements the paper makes.
Run from repro/data-scale. The probe outputs (P1 gate0, P2 cells, P3 ridge, P4 screening, P5 rest readout) are the files the
battery wrote on the Orin; probes.verdict() is the script's own function that joins them, so L0 recomputes verdict.json from them."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "scripts")
import probes  # noqa: E402  (numpy only at import time)

E = Path("expected")
SC = ["S1", "S3", "S4", "S5a"]
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond); print(("ok  " if cond else "FAIL") + " " + name)


def scrub(v):
    """verdict.json minus the machine paths it carries."""
    v = json.loads(json.dumps(v)); v.pop("dir", None)
    v.get("P1", {}).pop("source", None); v.get("P5", {}).pop("latents", None)
    return v


V = {}
for s in SC:
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / s; d.mkdir()
        for f in ("p1/gate0.json", "p2.json", "p3/results_ridge.json", "p4/w1a_summary.json", "p5.json"):
            (d / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy(E / s / f, d / f)
        V[s] = probes.verdict(d)
    check(f"{s}: verdict.json recomputed from the stored probe outputs", scrub(V[s]) == scrub(json.loads((E / s / "verdict.json").read_text())))

for s in ("S1", "S4", "S5a"):
    e = json.loads((E / s / "training_end.json").read_text())
    check(f"{s}: trained to the protocol's 81,068 steps (W1's)", e["kind"] == "end" and e["step"] == 81068 and e["stop_reason"] is None)

c2 = json.loads((E / "S3/c2.json").read_text())
check("C2: the probes reproduce W1's published values (8 of 8)", len(c2) == 8 and all(r["match"] for r in c2))

# P2: nothing leaves the action-only floor, nothing reaches the 0.8 bar
for s in SC:
    cells = V[s]["P2"]["cells"]
    check(f"{s} P2: every cell is below the action-only floor (CI of the difference below 0) and below 0.8",
          all(c["diff_ci95"][1] < 0 and c["model_top1"] < 0.8 for c in cells.values()) and V[s]["P2"]["largest_d_floor"] is None)

# P3: the three-part rule of the source protocol, recomputed from the bar's numbers
for s in SC:
    b = V[s]["P3"]["bar"]
    c1 = b["arm_ii_minus_max_i_iii"] >= 0.10 and b["both_cis_exclude_0"]
    c2_ = (b["arm_v_minus_iv"]["value"] >= 0.02 and b["arm_v_minus_iv"]["ci95"][0] > 0) or (b["gripper_v_minus_iv"]["value"] >= 0.05 and b["gripper_v_minus_iv"]["ci95"][0] > 0)
    c3 = b["random_init_arm_ii_minus_i"] < 0.5 * b["w1_arm_ii_minus_i"]
    fail = (b["w1_arm_ii_minus_i"] < 0.03) and not c2_
    branch = "PASS" if (c1 and c2_ and c3) else "FAIL" if fail else "NOT PROVEN"
    check(f"{s} P3: branch {branch} = stored {b['branch']} (conjuncts: gain {c1}, beyond joints {c2_}, random-init under half {c3})", branch == b["branch"] == "NOT PROVEN" and c1 and not c2_ and not c3)

# P4: bar is 5 of 8 failures at no more than 1 of 20 false interventions on the 50 episodes
for s in SC:
    g = V[s]["P4"]["g16_set"]
    check(f"{s} P4: no signal reaches 5 of 8 failures at 1 in 20 ({g['l2']['caught_failed']:.0f} / {g['cos']['caught_failed']:.0f} of 8)",
          all(g[k]["caught_failed"] < 5 and g[k]["failed_episodes"] == 8 and g[k]["success_episodes"] == 42 for k in ("l2", "cos")) and "FAIL" in V[s]["P4"]["verdict_source"])

# P5: the rest readout stays level with the colour judge at every scale
for s in SC:
    p = V[s]["P5"]
    check(f"{s} P5: AUROC {p['auroc']:.4f}, the paired difference to the judge has a CI that includes 0",
          p["n"] == 69 and p["diff_ci95"][0] < 0 < p["diff_ci95"][1])

# the curve file: training frames grow with the scale, and the values equal the verdicts
cv = json.loads((E / "curve.json").read_text())
fr = [cv[s]["train_frames"] for s in SC]
check("training frames rise with the scale", fr == sorted(fr) and fr == [15032, 53015, 275015, 1282161])
check("curve.json equals the recomputed verdicts", all(scrub({k: cv[s][k] for k in cv[s] if k != "train_frames"}) == scrub(V[s]) for s in SC))
c3 = json.loads((E / "c3_meta.json").read_text())["train_frames_native"]
check("training frames equal the dataset-meta counts of check C3", [c3[s] for s in SC] == fr)

# the splits: the held-out 25 are never in a training list, S4 and S5a add whole datasets
reg = json.loads(Path("../register/expected/split.json").read_text())
for s in ("S1", "S4", "S5a"):
    sp = json.loads(Path(f"splits/{s}/split.json").read_text())
    check(f"{s}: no validation episode in the training list", not set(sp["train_episodes"]) & set(sp["val_episodes"]) and len(sp["val_episodes"]) == reg["n_val"])
s1 = json.loads(Path("splits/S1/split.json").read_text())
check("S1 trains on 45 of the 157 episodes of S3", len(s1["train_episodes"]) == 45 and len(set(s1["train_episodes"])) == 45)
for s, n in (("S4", 1), ("S5a", 2)):
    sp = json.loads(Path(f"splits/{s}/split.json").read_text())
    check(f"{s}: {len(sp['train_episodes'])} bowls episodes plus {n} added dataset(s) = {len(sp['train_episodes']) + sum(x['n_episodes'] for x in sp['extra_train'])} episodes",
          len(sp["extra_train"]) == n and len(sp["train_episodes"]) == reg["n_train"])
s4, s5 = (json.loads(Path(f"splits/{s}/split.json").read_text()) for s in ("S4", "S5a"))
check("episode counts 157 + 653 = 810 and 810 + 2,099 = 2,909",
      157 + sum(x["n_episodes"] for x in s4["extra_train"]) == 810 and 810 + s5["extra_train"][1]["n_episodes"] == 2909)

print("L0 PASS" if ok else "L0 FAIL")
sys.exit(0 if ok else 1)
