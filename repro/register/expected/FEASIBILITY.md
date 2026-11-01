# Feasibility — "A small JEPA world model learns the arm, not the task": twelve pre-registered uses of one world model on a $200 robot

WS-091, 2026-11-01. Memo only (template `journal/prompts/2026-10-31_paper_feasibility_TEMPLATE.md`): nothing new was run
for it. Step 0, the checkpoint-load audit, is `AUDIT_W1_LOADS.md` (committed first): **26 rows trained, 1 random
(G11-A, already voided, carries no claim), 0 unknown**; one row (VLA-JEPA's own predictor) rests on indirect evidence.
Sources: `journal/EXPERIMENTS.md` (rows cited by line), WS-078's `PREREG_CHECK` classes, the gate folders, the papers
mirror.

## 1. Claim in one sentence

> **Reworded 2026-11-01 (owner go)** after `gaps/README.md`: the controls contradict "a frozen image encoder or the
> commanded joints do at least as well". Current claim and 4-page outline: `OUTLINE_4PAGE.md` — the encoder knows what
> happened (on rollouts W1 never saw; exploratory probes), the predictor does not know what will.

A LeWM-style JEPA world model trained on ~180 teleop demos of one SO-101 task ($3.30) predicts its own future
(open-loop error 0.31 vs copy-last 0.89) and reads a finished episode's outcome (AUROC 1.00), but fails every
pre-registered use that needs its latent to carry more than the arm's pose — ranking, progress, mid-episode monitoring,
contact, risk, planning, identifiability — and in each failure a frozen image encoder or the commanded joints do at
least as well; the claim is false if a scale point (3.6× the data) or a fair baseline turns one of those failures into
a pass.

## 2. Evidence in hand

| gate (EXPERIMENTS line) | date | n | result | bar | verdict | pre-reg. (WS-078) | load |
|---|---|---|---|---|---|---|---|
| G0 (46) | 10-11 | 25 val episodes | open loop K = 30: 0.31 vs copy-last 0.89; timing shifts 1.99 / 1.58; rank 24–30 | beats copy-last; ratios ≥ 1.3 | **pass** | same-commit | trained |
| C1 probe (47) | 10-11 | 81 rollouts (15 fail) | imagined 0.72, current 0.68; policy identity 0.76 | imagined beats current | not proven (policy confound) | bar-first | trained |
| G1b W1 (48) | 10-12 | 703 anchors | top-1 best 0.60; action-only 0.68–0.90, regret ≈ 0 (52) | top-1 ≥ 0.8 | **fail** | bar-first | trained |
| G1 probes (49) | 10-12 | 25 val episodes | arm-pose R² W1 0.57, **W2 (frozen DINOv2-S) 0.92** | > 0.57 | W2 pass | same-commit | trained |
| G1 W2 dynamics (51) | 10-12 | 703 anchors | W2 top-1 0.55–0.61 | ≥ 0.8 | **fail** | bar-first | trained |
| G6a (58) | 10-14 | val demos | monotonicity −0.32 to −0.50; done AUROC W1 0.97 (easy contrast); 32 % false stalls | ρ ≤ −0.8; AUROC ≥ 0.95 | T1 **fail**, T2 sanity | same-commit | trained |
| G10 (65) | 10-17 | 69 rollouts (not in W1's teleop training set, gaps/W1_TRAINING_SET.md) | readout at rest AUROC W1 1.000, W2 0.976; anticipation +0.179 over policy identity | ≥ 0.90; +0.10 | **pass** (retrospective) | same-commit | trained |
| G6b (66) | 10-18 | val demos | H-JEPA level 2 ρ −0.727; false stalls 52 % | ≤ −0.80; ≤ 10 % | **fail** | same-commit | trained |
| W1-A (81) | 10-19 | 50 (8 fail) | L2 surprise 2/8 at 2/42 (AUROC 0.86); drift 0/8; live reflex 42/42 successes flagged | ≥ 5/8 at ≤ 1/20 | **fail** | bar-first | trained |
| W1-B (82) | 10-19 | rollouts | 22.1 % premature, 31.1 % missed | ≤ 2 %, ≤ 10 % | **fail** | bar-first | trained |
| G11-B-cal (83) | 10-19 | 69 (25 fail) | 3/25 failures at 1/44 | recall ≥ 50 % | **fail** | bar-first | trained |
| W1-C1 (85) | 10-19 | 65 scored | imagined 0.70 < current 0.82 | ≥ 0.85, ≥ +0.05 | **fail** | bar-first | trained |
| W1-C2 (86) | 10-19 | 47 | successes 22.7 % flagged, failures 46 % | ≤ 1/20, ≥ 50 % | **fail** | bar-first | trained |
| W1-R (90) | 10-22 | 703 anchors × 5 seeds | regret 0.2631 vs W1 0.2630, action-only −0.0075 | ≤ 0.10 | **fail** | bar-first | trained |
| REAP v1 (95) / v2 (100) | 10-23 | 50 (8 fail) | v1 0.93 withdrawn (time leak); v2 3/8 at 5/42, AUROC 0.79 vs shuffled 0.71 | ≥ 5/8 at ≤ 1/20 | **fail** | bar-first | trained |
| §30 inverse dynamics (111) | 10-26 | teleop val + 69 | latent pair +0.748 over one frame, but random-init W1 1.254 vs W1 1.180; beyond state +0.007 | random-init < half W1 | **not proven** | same-commit | trained |
| LeJEPA (103, 106) | 10-27/28 | 8,447 val frames | W1 mean R² 0.565, anisotropy 0.639; W2′ 0.867 / 0.301 | R² < 0.80; ≥ 0.20 | B1–B3 met | bar-first | trained |
| SURVEYOR (104, 105) | 10-27 | 69 | τ 1.249 (paper 0.20); monitor 3/8 at 6/42 | τ ≤ 0.40; ≥ 5/8 at ≤ 1/20 | **fail** | bar-first | trained |
| HWM (107) | 10-28 | 700 anchors | no crossover: high level 0.76 vs W1 0.56 at 2 s | crossover in [1, 2] s | H1, H2 **fail** | bar-first | trained |
| RoboJEPA / V-JEPA 2.1 on device (54–56) | 10-13 | 30–60 clips | encoder 526 ms (2 views); ≤ copy-last + 12 % | ≤ 200 ms; ≥ 15 % | **no-go** | — | (pretrained) |

**Counts.** 20 rows: 3 passes (G0 prediction and G10 rest readout for W1; G1 probes for W2's pose reading), 1
met-as-predicted (LeJEPA: partial identifiability), 2 not proven (C1, §30), 14 fails / no-go. Pre-registration: 13
bar-first, 6 same-commit (G0, G1 probes, G6a, G10, G6b, §30) — **both positives are same-commit**, i.e. their bars cannot be shown
to predate the number from the repo alone. Blind labels: none (operator labels throughout). Statistic: the 10-19 and
later gates use the episode bootstrap (2,000); G0–G6b predate it and report point estimates — recomputable from stored
rows where they exist (`wm_train/results/g1_*`, `g6a/`, `g10/`, `g6b/`; $0).

## 3. Gaps, costed

| gap | what | resource | queued? |
|---|---|---|---|
| 1. **Scale point** | retrain W1 with the G0 recipe on `so101_cubes_bowls_20261015` (653 episodes, 3.6× the bowls set; or bowls + cubes pooled), then rerun G1b, LeJEPA R² (corrected protocol) and W1-A on the new checkpoint with the original bars | vast: G0 took ~9.2 h for 100 epochs on an RTX PRO 4000 at $0.28/h ($3.30 with retries) → ~33 h, **~$9–10**; the three reruns $0 on the Orin (~6 h) | new |
| 2. **Fair baseline for the rest readout** | G10 step 1 against the G9c masks (no learning, 0 FP) and against policy identity on the same 69, scored on rollouts W1 never saw (G10 is retrospective) | $0 offline for the 69; a held-out set needs fresh rollouts (≥ 20, part of any robot day) | new |
| 3. **Frozen-encoder / joints controls where missing** | have one: G1b (action-only, W2), G1 probes (W2), LeJEPA (W2′), §30 (W2′, random-init), HWM (post-hoc copy / mean). Missing: W1-A, W1-C1, SURVEYOR, G6a T1 with W2′ latents or commanded joints | $0 offline each, ~1 Orin day total | new |
| 4. **Random-encoder reruns** | none needed (G11-A carries no claim); harden the VLA-JEPA row with one logged key-set check | $0, minutes | new |
| 5. **G0 / G10 pre-registration** | both same-commit: say so in the paper, or re-run G10 step 1 with a bar committed first on fresh rollouts (gap 2) | with gap 2 | — |
| **total** | | **~$10 vast, ~3 Orin days, 0–1 arm day (fresh rollouts for gap 2)** | |

## 4. Related work

| paper | what it showed (on what) | different here |
|---|---|---|
| LeWorldModel 2603.19312 | end-to-end JEPA from pixels with SIGReg; planning in Push-T and sim tasks | same recipe on one real arm, 180 demos; planning use fails (G1b, HWM) |
| LeWM as reward 2607.25337 | temporal-distance JEPA for latent MPC, sim (Two-Room, OGB-Cube) | progress use fails here (G6a, G6b) |
| DINO-WM 2411.04983 | frozen DINOv2 patches + predictor plan in six sim environments | our W2 / W2′ is this design: reads pose (0.92) and still fails ranking |
| V-JEPA 2.1 2603.14482, RoboJEPA 2610.10515 | large video JEPAs, Franka planning; RoboJEPA scaling laws | too slow on the Orin (526 ms), ≤ copy-last + 12 % on our clips |
| VLA-JEPA 2602.10098 | JEPA world model inside a VLA, LIBERO / real Franka | its action head works on our arm (9/10, 20/20); its surprise untested (0 failures) |
| SaiVLA-JEPA 2610.38508 | future-latent head adds +0.03 points to control on 10,030 LIBERO-Plus episodes (sim) | independent agreement: a predictor's latent loss is not what control needs |
| ContactGuard 2608.13438 | imagined-latent probes beat current-latent probes on four grasp tasks | opposite here (W1-C1: 0.70 vs 0.82) |
| D-JEPA 2609.24749 | operator fixes global vs local prediction mismatch | its re-rank changes nothing here (W1-R) |
| HWM 2604.03208 | hierarchical latent world models beat flat ones at long horizons | no crossover here (HWM gate 0) |
| LeJEPA identifiability 2605.26379 | SIGReg latents identify state up to anisotropic transforms (sim) | holds here, partially (R² 0.565, anisotropy 0.639) |
| CF-JEPA 2610.00727 | on disk; not read for this memo | — |
| planning-limits-jepa 2609.39235, planning-jepa 2512.24497, HIJP 2610.23615, G-JEPA 2610.28483, physical-grounding 2608.06799 / 2609.03565, Platonic JEPAs 2610.39302 | **not on disk, not read here** | to add to Drive before the related-work section is written |

## 5. Objections

**(a) "Too small a model and too little data — of course it fails."** Answer: gap 1 (the scale point) turns one point
into a two-point curve, and the frozen-encoder controls (gap 3; already in for G1b, G1 probes, LeJEPA, §30) show a
pretrained 21M-image encoder fails the same uses. The paper's scope is "what people can train for a $200 arm", stated as
the condition.

**(b) "The encoder was random."** Answer: step 0 — 26 rows trained (25 strict loads; VLA-JEPA's non-strict LeRobot load with indirect
evidence), one random load (G11-A) voided the same day and carrying no claim, none unknown.

**(c) "Twelve uses is a fishing expedition."** Answer: each use is a published paper's claim (register entries), 13 of
the bars were committed before the number, and the two positives are reported beside the negatives — but both
positives are same-commit, so the paper must say that, and gap 2 gives the rest readout a bar-first rerun.

**(d) "Negative results do not transfer."** Answer: the diagnosis (the latent ≈ arm kinematics; the commanded joints or
a frozen encoder match it) and the one positive (rest readout) are testable on other rigs; the paper states them as
predictions with the measurement that would refute each.

## 6. Venue

Primary: a world-model or robot-learning workshop at ICLR 2027 or NeurIPS 2026–27 (dates to confirm on the workshop
pages before committing; 4 pages, double-blind). Fallback: TMLR (rolling, negative results in scope, no page limit).
Alternative: fold into §4 of the reproducibility paper (WS-069 / A). Recommendation on split vs fold: **split** — the
twelve uses are one coherent negative finding with a diagnosis, too long for a section, and the scale point makes it a
result rather than a list. arXiv first: after gap 1, not before. Endorsement: no contacts list in the repo.

## 7. Minimal vs full paper

4 pages: the twelve uses as one table (§2), the diagnosis figure (pose R² W1 vs W2′ vs joints; G1b W1 vs action-only),
the rest-readout positive with its fair baseline (gap 2), limitations. Needs gaps 2 and 4 (~$0, ~1 Orin day). 8 pages:
adds the scale point (gap 1, ~$10), the frozen-encoder controls for every negative (gap 3), and the per-use failure
analysis.

## 8. Risk of a null result

The "main claim" is itself mostly negative; the risk is the opposite: the scale point turns a negative into a pass (e.g.
G1b ≥ 0.8 with 653 episodes). Then the paper becomes a data-scale curve ("below N demos the latent is kinematics"),
which is a better paper. If the rest readout loses to the masks or to policy identity (gap 2), the one positive goes
and the paper is a clean negative with a diagnosis — publishable at a workshop or TMLR.

## 9. Recommendation

**go-after-X**, X = gaps 2 and 4 ($0, ~1 Orin day): if the rest readout keeps its value against the masks and policy
identity, write the 4-page version; run the scale point (~$10) for the 8-page one. Total ≈ $10 vast, ~3 Orin days,
≤ 1 arm day. Earliest: a workshop draft two weeks after the go (the gaps are offline); TMLR version after the scale point.

## 10. Skeleton

Titles: (1) *A small JEPA world model learns the arm, not the task*; (2) *Twelve uses of one world model on a $200 arm*;
(3) *What 180 demonstrations teach a JEPA world model*.

Abstract (≤ 150 words): We trained a LeWM-style JEPA world model on 180 teleoperated demonstrations of one SO-101 task
for $3.30 and put it through twelve pre-registered uses taken from published papers. It predicts its future (open-loop
error 0.31 vs 0.89 for copying the last frame) and reads the outcome of a finished episode (AUROC 1.00; against
deterministic masks and policy identity: [to run]). It fails every use that needs more than the arm's pose: ranking
candidate chunks (0.60 vs 0.68–0.90 for the commanded joints alone), progress, mid-episode monitoring (2/8 failures),
contact, risk, planning across horizons, and the latent recovers the joints with R² 0.565. A frozen DINOv2 encoder
reads the pose better (0.92) and fails the same uses. At 3.6× the data: [to run]. On a $200 arm, a JEPA world model's
latent is kinematics.

Sections: 1 Intro · 2 The model and the rig · 3 Twelve uses, one table · 4 The diagnosis: pose, joints, frozen
encoders · 5 The positive: reading the outcome at rest · 6 Scale point [to run] · 7 Related work · 8 Limitations.

Figures: (1) the twelve uses as a bar-vs-result chart — EXPERIMENTS rows (today); (2) pose R² per joint W1 vs W2′ —
`paper-vs-hardware/posts/2026-10-29-paper-vs-hardware-lejepa/expected/gate0.json` (today); (3) G1b top-1, W1 vs W2 vs
action-only — `g1b_range_w1.json`, `g1_investigation.json` (today); (4) the scale curve [to run].
