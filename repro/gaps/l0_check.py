#!/usr/bin/env python3
"""paper-jepa-world-model L0 (both repro folders): recompute the counts and orderings the paper states from expected/.
Run from repro/gaps or repro/register."""
import json
from pathlib import Path

E = Path("expected")
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond); print(("ok  " if cond else "FAIL") + " " + name)


if (E / "controls.json").exists():
    c = json.loads((E / "controls.json").read_text())
    check("n = success + failure", c["n"] == c["success"] + c["failure"])
    g = {k.split(" ")[0]: v["auroc"] for k, v in c["gap2"].items()}
    check("rest: W1 > W2' > joints > policy", g["W1"] > g["W2'"] > g["joints"] > g["policy"])
    j = c["gap2"]["G9c-3 judge score s (no learning)"]["ci95"]; w = c["gap2"]["W1 at judge frame (LOSO probe)"]["ci95"]
    check("rest: W1 and judge CIs overlap", w[0] <= j[1] and j[0] <= w[1])
    a = {k.split(" ")[0]: v["auroc"] for k, v in c["w1c1"].items() if isinstance(v, dict)}
    check("anchor: W1 > W2' > joints", a["W1"] > a["W2'"] > a["joints"])
    for k, v in c["w1c1"].items():
        if isinstance(v, dict):
            s = v["auroc_oof_per_seed"]
            check(f"anchor {k}: auroc = mean of seeds", abs(sum(s) / len(s) - v["auroc"]) < 6e-5)
    k = json.loads((E / "vla_jepa_keyset.json").read_text())
    check("VLA-JEPA world-model keys: none missing", k["world_model_keys_missing"] == 0)
else:
    t = (E / "FEASIBILITY.md").read_text().split("## 2.", 1)[1].split("## 3.", 1)[0]
    rows = [l for l in t.splitlines() if l.startswith("| ") and not l.startswith("| gate")]
    v = [l.split("|")[6].lower() for l in rows]
    cls = ["np" if "not proven" in x else "met" if "met" in x else "pass" if "pass" in x and "fail" not in x else "fail" for x in v]
    stated = (20, 3, 1, 2, 14)
    got = (len(rows), cls.count("pass"), cls.count("met"), cls.count("np"), cls.count("fail"))
    check(f"register rows / pass / met / not proven / fail = {got} as stated in §2 Counts {stated}", got == stated)
    p = [l.split("|")[7].strip() for l in rows]
    check("pre-reg: 13 bar-first, 6 same-commit", (p.count("bar-first"), p.count("same-commit")) == (13, 6))
    s = json.loads((E / "split.json").read_text())
    check("split: n_train + n_val episodes listed", len(s["train_episodes"]) == s["n_train"] and len(s["val_episodes"]) == s["n_val"]
          if isinstance(s["train_episodes"], list) else s["train_episodes"] == s["n_train"])
print("L0 PASS" if ok else "L0 FAIL")
