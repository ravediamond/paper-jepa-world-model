# 4-page outline — "A small JEPA world model knows what happened, not what will"

WS-091, 2026-11-01. Claim reworded on the owner's go (2026-11-01 afternoon) after the controls in `gaps/README.md`
contradicted FEASIBILITY.md §1 ("a frozen DINOv2 or the commanded joints do as well"). This replaces FEASIBILITY.md §7's
minimal version and §10's title / abstract for the 4-page paper. The scale point is WS-106's, not here. Sources: the
gate rows of FEASIBILITY.md §2 (EXPERIMENTS lines), `gaps/controls.json` (69 rollouts, 44 / 25, 13 sessions),
`AUDIT_W1_LOADS.md`. No number below is new; no bar is written here.

**Title.** *A small JEPA world model knows what happened, not what will* (alt.: *Twenty pre-registered uses of one
world model on a $200 arm*).

**Claim (one sentence).** On ~180 teleop demos of one SO-101 task, a LeWM-style JEPA encoder carries the episode's
outcome — at rest (AUROC 0.95) and at the grasp anchor (0.82), above a frozen DINOv2-S (0.87 / 0.71) and the commanded
joints (0.76 / 0.53) — but every pre-registered use that needs its *predictor* to turn that into a forecast (ranking,
progress, surprise, imagined contact, risk, planning) fails: the predictor, not the encoder, is the weak link.
**Out-of-sample for the encoder:** W1 was trained on teleop only; the 69 policy rollouts are a separate dataset it never
saw (`gaps/W1_TRAINING_SET.md`). The readouts are LOSO probes on those 69, exploratory (no bar before the data).

**Abstract (≤ 150 words, draft).** We trained a LeWM-style JEPA world model on 180 teleoperated demonstrations of one
SO-101 task for $3.30 and ran it through twenty pre-registered uses taken from published papers. Its encoder knows what
happened: a linear probe reads the outcome of a finished episode at AUROC 0.95, level with a hand-built colour judge
(0.94) and above a frozen DINOv2 (0.87), the commanded joints (0.76) and policy identity (0.59); at the grasp anchor the
current latent reads the grasp outcome at 0.82 (DINOv2 0.71, joints 0.53). It does not know what will happen: the
imagined latent reads the grasp worse than the current one (0.70 vs 0.82), and ranking, progress, surprise monitoring,
risk and multi-scale planning all miss their bars. The rollouts were never seen by the encoder. One rig, one model, one data scale.

## Sections (4 pages)

1. **Intro** (½ p). One small world model, one $200 arm, twenty pre-registered uses; papers report the positive use,
   we report all of them. Contribution: the register, and a split of the result into *encoder* (outcome is there) vs
   *predictor* (forecast is not). **Caveat stated here:** the encoder readouts are exploratory LOSO probes on 69 rollouts
   W1 never saw (teleop-only training), with operator labels, and two of the positives have same-commit bars.
2. **Setup** (½ p). W1 (G0 recipe, ~180 teleop episodes, $3.30); controls W2′ (frozen DINOv2-S, mean-pooled, top +
   wrist) and commanded joints; G9c-3 colour judge (no learning). Load audit: 26 trained, 1 random (G11-A, voided),
   0 unknown; VLA-JEPA world-model keys 351 / 351 (`gaps/vla_jepa_keyset.json`).
3. **The register** (1 p, one table = FEASIBILITY.md §2, columns *use · bar · result · verdict · pre-reg.*), grouped:
   - *encoder reads* — G10 step 1 rest readout (pass, same-commit), W1-C1 current latent 0.82, G1 / LeJEPA pose R²
     (W1 0.565, W2′ 0.867: the encoder is not mainly kinematics, nor a better pose reader);
   - *predictor forecasts* — G0 open-loop 0.31 vs copy-last 0.89 (pass: it predicts its own latent), then every use
     of that prediction: G1b ranking 0.60 (bar 0.8), W1-R regret, G6a / G6b progress, W1-A surprise 2/8 at 2/42,
     W1-B, G11-B-cal, W1-C1 imagined 0.70 < current 0.82, W1-C2, REAP v2, SURVEYOR, HWM — all fail.
   Counts: 3 pass, 1 met, 2 not proven, 14 fail / no-go; 13 bar-first, 6 same-commit.
4. **What the encoder knows** (¾ p). The gap-2 table (rest readout: W1 0.9545 [0.902, 0.991], G9c-3 0.9432, W2′ 0.8655,
   joints 0.7573, policy 0.5882) and the grasp-anchor row (0.8199 / 0.7067 / 0.5305, 65 scored, 21 fail). Reading:
   W1 ≈ the colour judge at rest (CIs overlap), above the frozen encoder and the joints at both points. Labelled
   exploratory: no bar was committed before these controls.
5. **Where the predictor fails** (¾ p). The cleanest contrast is W1-C1: same encoder, same anchor, current latent 0.82
   vs imagined 0.70 (bar ≥ 0.85 and ≥ +0.05 over current). G1b: the commanded joints alone rank chunks better
   (0.68–0.90) than W1 (0.60). SURVEYOR: W1's τ 1.249 fails S1 and S2; W2′ and joints pass S2 (scale), all fail S1.
   Objection to answer: G10 step 2 (imagined latent +0.179 over policy identity) is a predictive pass, but same-commit
   and retrospective, and the bar-first reruns of the same idea (W1-C1, G11-B-cal) fail.
6. **Limitations** (½ p). **Exploratory readouts**: rollouts W1 never saw, but 69 of them, operator labels, probes
   chosen after the data; fresh rollouts with a bar committed first are the test. One rig, one model, one data scale (the 653-episode point
   is WS-106). Controls not run: W1-A surprise and W1-C1 imagined arm with W2′ / joints (need a predictor per
   feature), G6a T1 (needs a reimplementation). G0 and G10 bars are same-commit.

## Figures
1. The register as bar-vs-result, grouped encoder / predictor (FEASIBILITY.md §2).
2. Outcome AUROC at rest and at the grasp anchor, W1 vs W2′ vs joints vs judge vs policy, with CIs (`gaps/controls.json`).
3. W1-C1 current vs imagined (FEASIBILITY.md §2 W1-C1 row; `gaps/controls.json` → w1c1).

## Open before a draft
- Fresh rollouts with a bar committed first, to take the encoder claim out of exploratory (arm day; not queued).
- Optional $0 controls: W1-A / W1-C1 imagined with the W2′ predictor (`w2prime/weights_best.pt`); G6a T1 on
  `projects/lejepa-identifiability/out_v2/latents.npz`. Queue only on the owner's word.
