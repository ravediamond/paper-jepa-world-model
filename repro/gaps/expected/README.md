# WS-091 gaps 2–4 (owner go 2026-11-01): fair baselines and frozen-encoder / joints controls

**Exploratory.** No bar was committed for these comparisons before the data. The original gate bars are quoted
only where a rerun uses the same script. Data: the 69 rollouts `ravediamond/so101-bowls-policy-rollouts` @ the
`data/FACTS.json` → rollouts revision (44 success / 25 failure, 13 sessions; labels from `meta/rollout_labels.csv`).
W1 was trained on teleop only; these rollouts are out-of-sample for it (`W1_TRAINING_SET.md`); the probes are LOSO on the 69.
All numbers: `controls.json` (from `analyze_controls.py`). Features: `extract_controls.py` (log `runs/WS-091/extract.log`).
- W1 = the cached strict-load latents `outputs/reap/latents/` (W1-A run).
- W2′ = frozen DINOv2-small, mean-pooled patch tokens, top + wrist.
- joints = the recorded commanded action.

**Reproduction check.** On the cached W1 latents the pipelines reproduce the published values: W1-C1 current latent 0.8199
(published 0.820, `shared/gate/data/2026-10-19_w1c/README.md`) and SURVEYOR τ 1.2488 (published 1.249,
`projects/surveyor/results/gate0.json`).

## Gap 2: rest readout (G10 step 1) against fair baselines, same 69 episodes

Readout at the G9c-3 judge frame (`projects/surveyor/results/judge.csv`), scored against the operator labels. Probe:
leave-one-session-out L2 logistic (StandardScaler, C = 1, balanced), as `g10_eval.fit_loso`. 95 % CI: episode
bootstrap, 2,000 resamples.

| readout | learning | AUROC | 95 % CI |
|---|---|---|---|
| W1 latent | LOSO probe | **0.9545** | [0.902, 0.991] |
| G9c-3 deterministic judge score s | none (HSV masks) | 0.9432 | [0.892, 0.988] |
| W2′ (frozen DINOv2-S) | LOSO probe | 0.8655 | [0.777, 0.941] |
| commanded joints | LOSO probe | 0.7573 | [0.608, 0.888] |
| policy identity | LOSO probe | 0.5882 | [0.408, 0.752] |

Reading: at rest, W1 is about as good as the no-learning colour judge (the CIs overlap) and above both the frozen
encoder and the joints. Policy identity is near chance under LOSO. G10's 1.000 was scored against judge labels on 87
rollouts. Here, against operator labels on the 69, W1 gets 0.95.

## Gap 3: frozen-encoder / joints controls (same scripts, features swapped)

| gate (original bar) | W1 | W2′ | joints | reading |
|---|---|---|---|---|
| W1-C1 current feature at the anchor, AUROC (bar for the *imagined* latent: ≥ 0.85 and ≥ +0.05 over current) | 0.8199 | 0.7067 | 0.5305 | W1's current latent carries grasp-outcome information that the joints do not (65 scored, 21 fail) |
| SURVEYOR gate 0 τ (S2: τ ≤ 0.40; S1: τ < p10 of non-equivalent) | 1.2488 (S1 ✗, S2 ✗) | 0.3601 (S1 ✗, S2 ✓) | 0.0265 (S1 ✗, S2 ✓) | S2 depends on scale (relative distance). S1, the separation bar, fails for every feature. Eq-vs-non-eq AUROC: W1 0.712, W2′ 0.833, joints 0.866 |
| W1-A L2 surprise | 2/8 at 2/42 (published) | not run | not run | needs a predictor for each feature (the W2′ predictor `w2prime/weights_best.pt`, or a joints dynamics model). Not the same script with features swapped |
| G6a T1 monotonicity | −0.50 (published) | not run | not run | G6a's sub-goal bank is not in the cached inputs. `projects/lejepa-identifiability/out_v2/latents.npz` holds h_w1 / h_w2p / z for the 25 val episodes, so this needs a reimplementation |

W1-C1's imagined-latent arm (needs each feature's own predictor) was not rerun. Only the current-feature baseline was.

## Gap 4: VLA-JEPA key-set check

Run 2026-11-01 (dispatcher session, `IMAGE=lerobot-jetson:rc LEROBOT_SRC=image shared/gate/jetson_run.sh --python
journal/papers/jepa-world-model/gaps/vla_jepa_keyset.py <snapshot c65c98b> …`): result in `vla_jepa_keyset.json` —
module 1,613 keys, file 1,612; **world-model keys 351, missing 0**; unexpected 0; the one missing key in the whole module
is `model.qwen.model.model.language_model.embed_tokens.weight` (the VLM's input embedding, not used by the world
predictor; most likely tied weights not saved separately — not checked here). The AUDIT row F stays **trained** for
the world predictor, now on a logged key set rather than indirect evidence.

## What this changes in FEASIBILITY.md

The diagnosis "a frozen DINOv2 or the commanded joints do as well" does **not** hold on these two uses. At rest and at
the grasp anchor, W1 > W2′ > joints. On SURVEYOR, W2′ and joints separate the pairs better (eq-vs-non-eq AUROC).
"The latent is kinematics" is therefore weaker than §1 states. The paper's claim needs rewording before the outline is final.

## 4-page outline

Superseded by `../OUTLINE_4PAGE.md` (claim reworded on the owner's go, 2026-11-01: "knows what happened, not what will").

## §5 exploratory lines (added 2026-11-02)

- `gap5_pooled_rerun.json` — road-to-robotics `projects/jepa-distill-edge/results/gap5_pooled_rerun.json` @ fb68283
  (pooled §30 term, teleop val, paired episode bootstrap): trained W1 vs its random init. `gap5_floor_stability.md`:
  the README excerpt on the random-init term's machine dependence (BLAS / ill-conditioned ridge).
- `M1_diag_g0_repro.json` — road-to-robotics `projects/jepa-surprise-receptive-field/results/M1_diag_g0_repro.json`
  @ fb68283 (WS-116 M1 diagnosis): the surprise scorer reproduces G0's K=30 corruption ratios (`by_k/30` vs `g0_k30`).
- `M1_diag_note.md` — the README excerpt stating M1's class-5 step-2 AUROC (0.775) and that it stands as measured.
