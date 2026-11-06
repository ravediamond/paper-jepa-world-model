# PROTOCOL — the small-data curve of a JEPA world model (WS-106)

WS-106, 2026-11-01. **Frozen before any number**: no model in §2 except S3 (= W1) has been trained; no probe in §3 has
been run on any new checkpoint. Sources: the feasibility memo `journal/papers/jepa-data-scale/FEASIBILITY.md` (§2 its
nine corrections, §3 its gaps), the prompt `journal/prompts/2026-11-01_paper_jepa-data-scale.md` ("Design detail"), the
owner's go of 2026-11-01 afternoon (item log, `journal/workstack/QUEUE.md` WS-106), WS-091's gap 2 protocol
(`journal/papers/jepa-world-model/gaps/README.md`). Every bar in §5 is copied from its original protocol; the floor
comparisons (§4, §5 "floor rule") are the owner's and the prompt's, worded here; the wording choices that decide a verdict
and were not given are listed in §9 and block training until the owner settles them.

> **Amendment 3 (owner, 2026-11-02, before any S4 / S5a number; full text `AMENDMENT_2026-11-01.md` §Amendment 3): the C4 rate rule is (a), native 20 Hz, no resampling.**
> S5a is reported as **"scale + ArmnetBench at native 20 Hz"**; S4 is the clean same-rate point (our data at 30 Hz). The 224-px transcode of ArmnetBench is resolution only. No bar, threshold or decision rule above changes.

## 1. Question

As the same LeWM-style JEPA recipe is trained at equal gradient steps on 45 → ~2,900 SO-101 episodes, does any of the
three predictor uses that failed at ~180 demos (G1b ranking, §30 action beyond state, W1-A screening) leave its floor,
and how do the two encoder readouts (joints R², the rest readout of a finished episode's outcome) move — i.e. does the
curve separate "knows what happened" from "knows what will happen" (`journal/papers/jepa-world-model/OUTLINE_4PAGE.md`)?

## 2. Data, scales, training

**Fixed held-out set at every scale:** the 25 val episodes of `projects/vla-jepa-world-model/wm_train/split.json`
(`so101_bowls_all_trimmed` @ `4e5c6fda`, split seed 1234, 157 train / 25 val). They are never in any training set.
The 69 rollouts `ravediamond/so101-bowls-policy-rollouts` (`data/FACTS.json` → `rollouts`, 44 success / 25 failure)
and their 50-episode subset (W1-A, 42 ok / 8 fail) are never in any training set either (correction 9 resolved:
W1's config trains on `so101_bowls_all_trimmed` only, `wm_train/config/lewm_mv_so101.yaml` `data.repo_id`; the
"seen as unlabelled video" line in `gaps/README.md:6` has no run log behind it and is contradicted by
`shared/gate/data/2026-10-19_w1_screening/README.md:16`).

**Scales (nested; memo correction 3):**

| scale | training episodes | rule |
|---|---|---|
| S1 | 45 | first 45 of a `numpy.random.default_rng(0)` permutation of the 157 train episodes |
| S2 | 90 | first 90 of the same permutation (S1 ⊂ S2) |
| S3 | 157 | the 157 train episodes = W1 as published (no retrain) |
| S4 | 810 | S3 ∪ all 653 `so101_cubes_bowls_20261015` episodes |
| S5a | 2,909 | S4 ∪ all 2,099 ArmnetBench rollouts (`armnet/armnetbench_v01_lerobot_so101` @ `1db69c82`), success and failure |
| S5b | S4 + successes | S4 ∪ the ArmnetBench rollouts labelled success that are not teleop (memo §3 gap 1: 915 − 400 = 515; count re-read in check C3) |

S5a / S5b add a second rig (other lab, other cameras and fps; memo correction 4): the paper states that S5 changes data
and rig together. The x-axis of every figure is training **frames** (counted from each dataset's meta, check C3).

**Recipe.** G0's: `wm_train/config/lewm_mv_so101.yaml` unchanged (cameras, resolution, frame stride, loss = prediction
+ SIGReg, hyper-parameters) except the dataset list. Training seed 3072 (the config's) at every scale; extra seed 0 at
S1 and S5a (memo §3 gap 3, primary budget: one extra seed).

**Training budget and checkpoint rule (owner 2026-11-01):** fixed gradient steps — every training runs exactly N steps,
N = W1's final step count (≈ 81,100 = 811 steps/epoch × 100, `wm_train/README.md:119`; exact N read in check C1), and
the **final** checkpoint at step N is the one probed (W1 is itself the final checkpoint, `weights_final.pt` @
`d0336db6`, `journal/papers/jepa-world-model/AUDIT_W1_LOADS.md:31`). No best-val selection. One extra point:
**S4 at fixed epochs** (100 epochs over S4, final checkpoint), reported beside the S4 fixed-steps point as "more compute".

**Controls (trained under the same step and checkpoint rule):**
- **W2′ per scale**: frozen DINOv2-S encoder + predictor trained on the same scale's episodes (S1, S2, S3, S4, S5a,
  S5b). The published W2′ (`w2prime/weights_best.pt` @ `b5bd6036`) is best-val at step 6,000, so it is **not** the S3
  point; it is retrained at S3 under this rule and the published one is reported as a reference only.
- **Random-init encoder, frozen** + predictor trained at S1, S3, S5a (memo gap 5). For encoder-only probes the random
  encoder is one checkpoint (no training).
- Encoder-only probes (R², rest readout, §30) on W2′ and the random encoder are scale-independent by construction (memo
  correction 6): computed once, drawn as flat lines.

**Probe data are fixed across scales**: every probe fits and scores on the same episodes at every scale; only the
checkpoint changes.

## 3. Metrics (one script `projects/jepa-data-scale/probes.py`, run on every checkpoint)

Bootstrap everywhere: episode bootstrap, 2,000 resamples, `numpy.random.default_rng(0)`, 95 % percentile CI, unit of
resampling = episode; comparisons against a floor are **paired** (the same resampled episodes for model and floor).

| # | probe | axis | data | metric (as in the original script) |
|---|---|---|---|---|
| P1 | joints R² | encoder | 157 train (every 3rd frame) / 25 val (all frames) | ridge h → z, z = the 6 joint positions in `observation.state`, α by 5-fold CV grouped by episode (train only); mean R² over joints and anisotropy (max − min per-joint R²) — `projects/lejepa-identifiability/PROTOCOL.md` (corrected protocol, `scripts/gate0_analyze.py`) |
| P2 | G1b ranking | predictor | 25 val episodes, 703 anchors | top-1 of 8 candidate chunks (true, 4 cross-episode, ±15 shifted, held); score = L2 of last imagined latent vs z_{t+d}; correct = the true chunk has the strictly lowest score (ties count against it, `G1_PROTOCOL.md:36-38`); cells d ∈ {5, 10, 15, 20, 30} at H = d — `wm_train/g1b_range.py` (not the `g1_investigation.json` protocol, whose W1 value 0.6117 the LeJEPA protocol cites) |
| P2-floor | action-only | — | same anchors and candidates | top-1 of the action-only ranker of `wm_train/G1_PROTOCOL.md` "G1 diagnosis" (line ~211), same cells |
| P3 | §30 beyond-state | encoder | teleop val anchors (25 val ep.), ridge fit on the 157 train | inputs (i)–(v) (`shared/gate/data/2026-10-24_inverse_dynamics/README.md:21-26`); primary cell as the source (line 51): this scale's encoder, both views concatenated, k = 5, ridge, teleop val; "arm" = mean residual R² over the 5 arm joints, "gripper" = open/close accuracy; differences paired by episode. k ∈ {1, 5, 15}, MLP reported beside, not gated |
| P4 | W1-A screening | predictor | the 50 episodes (8 fail); the 69 as a second column | as `projects/vla-jepa-world-model/V1.0_REVIEW.md:241-248`: anchor every 25 frames, context = anchor + 2 frames before with recorded actions, imagined to k = 25; per anchor at k ∈ {15, 20, 25} L2 surprise ‖z − ẑ‖ and cosine drift; episode score = max over anchors of the mean over the three k; τ = smallest threshold with ≤ 1/20 successes flagged (in-sample, optimistic, as original; "flagged" and the 1/20 count exactly as `scripts/run_w1_offline_screening.py`, which gives 2/42 at S3); failures caught at τ. Live reflex reported, not re-tuned |
| P5 | rest readout | encoder | the 69 rollouts, operator labels | latent at the G9c-3 judge frame (`projects/surveyor/results/judge.csv`); leave-one-session-out L2 logistic (StandardScaler, C = 1, balanced), as `g10_eval.fit_loso`; AUROC of the pooled out-of-fold scores, CI as `journal/papers/jepa-world-model/gaps/analyze_controls.py` `boot` (episodes resampled over the pooled out-of-fold scores, no refit) — gap 2 |

## 4. Baselines and floors (same set, same episodes)

| probe | floor (the comparison the curve is read against) | also reported |
|---|---|---|
| P1 | none (descriptive axis "learns the arm") | W2′ and random encoder (flat); B1 / B2 of the LeJEPA protocol per scale |
| P2 | **action-only top-1, same cells** (owner 2026-11-01: primary comparison) | the 0.8 bar; W2′ per scale; random encoder |
| P3 | built into the original rule: (iv) joints and the random-init conjunct | W2′ (flat) |
| P4 | W2′ at the same scale (prompt "Design detail": floor = W2′ or action-only; no action-only screening exists) | random encoder at S1 / S3 / S5a |
| P5 | **open (O1)** — gap 2 had no floor; its baselines: G9c-3 judge score, W2′, commanded joints, policy identity | the G10 bar 0.90 (against judge labels in G10; here operator labels) |

Identity / length null: P2's "held" candidate and the action-only ranker (identity of the commanded chunk); P5's policy
identity; P3's (iii) random-other-episode input.

## 5. Bars (verbatim from the original protocols; reported per scale as pass / fail / not proven)

- **P2 (G1b).** "The largest d in {5, 10, 15, 20, 30} with accuracy >= 0.8 at H = d" — `wm_train/G1_PROTOCOL.md:41-42`.
  **Headline (owner 2026-11-01): the action-only floor is the primary comparison, the 0.8 bar reported alongside.**
  Floor rule, same form as the original: the largest d in {5, 10, 15, 20, 30} at H = d where the paired CI of
  (model top-1 − action-only top-1) lies above 0; "none" if no d. *Wording to confirm (O2).*
- **P3 (§30).** "PASS (the latent pair carries the action): arm (ii) − max((i), (iii)) ≥ 0.10 with CI excluding 0,
  and [arm (v) − (iv) ≥ 0.02 with CI excluding 0, or gripper (v) − (iv) ≥ 5 points with CI excluding 0], and the
  random-init W1's arm (ii) − (i) gain is less than half of W1's." "FAIL: arm (ii) − (i) < 0.03 or its CI includes 0,
  and (v) is not better than (iv) on arm or gripper (both CIs include 0)." "NOT PROVEN: anything else." —
  `shared/gate/data/2026-10-24_inverse_dynamics/README.md:51-54` (`35f4600`). The three-part rule, not the first
  conjunct alone (memo correction 1). "W1" reads as this scale's model; the random-init term is the same untrained
  encoder at every scale (the one the source used).
- **P4 (W1-A).** "one W1 signal flags ≥ 5/8 failures at ≤ 1/20 false interventions" —
  `projects/vla-jepa-world-model/V1.0_REVIEW.md:249-250`. "One signal" = L2 surprise or cosine drift, as in the source
  (the live reflex reported only); "W1" reads as this scale's model.
- P2 and P4's sources define pass / fail only; no "not proven" class is added here.
- **P5 (rest readout).** G10: "Pass (per encoder): AUROC >= 0.90 against the judge labels" —
  `wm_train/G10_PROTOCOL.md:44-45`; reported alongside only (gap 2 scores operator labels, not judge labels).
- **P1.** LeJEPA B1 "W1 mean R²(h→z) < 0.80", B2 "anisotropy ≥ 0.20" — `projects/lejepa-identifiability/PROTOCOL.md:14-15`;
  reported per scale, not a claim bar.

**Decision rule (prompt "Design detail", memo §1).** No bar on the curve's shape: it is descriptive. For each probe the
paper reports the smallest scale at which its CI clears its floor (§4), or "none". Headline: whether P2–P4 leave their
floors by S5a. The memo's claim ("predictor uses stay at their floor") is **false** if, at any scale, one of P2–P4
clears both its original bar and its floor (P3: its rule includes the floor). Seeds: a probe "leaves the floor" at S1
or S5a only if it does at both seeds.

## 6. Checks before numbers

- **C1** N: W1's final step count from its run log / checkpoint (not the estimate); recipe diff = dataset list only.
- **C2** Reproduction: `probes.py` on W1 reproduces the published S3 values (P1 mean R² 0.5654 / anisotropy 0.6393,
  `projects/lejepa-identifiability/results/gate0_corrected_2026-10-28.json`; P2 best cell 0.6046,
  `wm_train/results/2026-10-12_g1b_range_w1.json`; P3 arm (ii)−max +0.748 and (v)−(iv) +0.007,
  `shared/gate/data/2026-10-24_inverse_dynamics/`; P4 2/8 at 2/42, `shared/gate/data/2026-10-19_w1_screening/`;
  P5 0.9545, `journal/papers/jepa-world-model/gaps/controls.json`). A mismatch stops the run.
- **C3** Data: FACTS.json entries (revision, episodes, frames) for `so101_bowls_all_trimmed`, `so101_cubes_bowls_20261015`
  and ArmnetBench (none exist today); the S5b success count; the S1 / S2 episode lists committed before training.
- **C4** ArmnetBench mapping (memo gap 6): camera map to front_top + wrist, 20 → 30 Hz stride rule, degrees → our joint
  units; 2 episodes decoded and plotted against one of ours. Committed before S5 training.
- **C5** No held-out episode (25 val, 69 rollouts) in any training list (scripted check, logged).
- **C6** One hand-computed example per probe on 2 episodes.
- **C7** The random-init encoder used by §30 is located (file, seed) and reused; if it cannot be found, stop and ask.
- Batch size, optimiser and schedule come from the config unchanged at every scale (C1's diff); the S1 / S2 final
  checkpoint is probed even if it overfits (the rule above), with the val loss curve logged beside it.

## 7. Deviations

(empty)

## 8. Exploratory

Everything not in §5: P1's curve shape, W2′ / random-encoder lines, S4 fixed-epochs vs fixed-steps, S5a vs S5b, k ≠ 5
in P3, H ≠ d in P2, regret, the published W2′ reference. Any choice made after a number is labelled exploratory.

## 9. Open items (owner, before any training)

- **O1** P5 floor: which baseline the rest readout's "leaves the floor" is read against (judge score, W2′, joints).
- **O2** P2 floor rule wording in §5 (the original "largest d" form with the floor in place of 0.8).
- **O3** P4's floor comparison: e.g. the paired CI of (model − W2′) failures caught, each at its own in-sample τ —
  with 8 failures the CI is coarse; and whether the floor is W2′'s same signal or its best signal.
- **O4** Multiplicity and matching: P2 has 5 cells, P3 3 k, P4 2 signals, 6 scales, 2 seeds at S1 / S5a. Must the bar
  and the floor be cleared at the same cell (P2: same d)? Does a single-seed scale (S2, S3, S4, S5b) count for the
  falsification rule, given the both-seeds rule at S1 / S5a? Any correction for the cells? "Smallest scale" when a
  probe clears at a scale but not at a larger one: report as measured (non-monotone), or the smallest scale from which
  every larger one also clears (as G1_PROTOCOL.md:43 does for d)?
- **O5** S5b and S4-fixed-epochs: in the headline ("by S5") or exploratory beside it (§8 lists them exploratory)?
- **O6** Seeds of the control predictors: W2′ at the published W2′ seed 3073 at every scale; random-encoder predictor
  seed; seed-0 reruns of controls at S1 / S5a or not.

## 10. Cost (memo §3; nothing launched)

W1 trainings S1, S2, S4, S5a, S5b ≈ 5 × $3.30; S4 fixed epochs ≈ $17; seeds 2 × $3.30; W2′ 6 × $0.65; random encoder
3 × $0.65; ArmnetBench preprocessing ≤ $1.5 — vast needs `go vast WS-106`. Probes ≈ 4 Orin days, 0 arm days.
