# PROTOCOL_CONFIRM — the encoder/predictor split on fresh rollouts (WS-135 Part 2)

WS-135 · 2026-11-02 · **frozen before any number**: committed before any fresh rollout is recorded. From this commit on,
any change is a numbered deviation in §7 with its reason and date; anything chosen after data is exploratory.
Bars are copied from the item prompt (`road-to-robotics/journal/prompts/2026-11-02_paper_fix_jepa_world_model.md`,
Part 2); this file writes no bar of its own.

## 1. Question
On rollouts recorded after this commit, does the final-frame latent of our world model (W1) read the outcome above the
frozen DINOv2-S control, and does the current latent at the grasp anchor read it better than the imagined one?

## 2. Data
- **Fit set (old):** the 69 labelled rollouts of `ravediamond/so101-bowls-policy-rollouts` at the revision in
  `road-to-robotics/data/FACTS.json` key `rollouts` (the set of `repro/gaps/expected/controls.json`, n = 69).
- **Test set (fresh):** 30 new rollouts on the SO-101 rig with the same SmolVLA checkpoints as the fit set; target
  ≥ 8 failures; if fewer than 8 failures after 30, continue and stop at 40 whatever the count (prompt Part 2).
  Rows to `repro/gaps/expected/confirm_rows.jsonl`, one per episode: episode, policy checkpoint, operator label, then
  the scores of §3.
- **Labels:** the operator's success / failure label, written at rest **before** any probe score of that episode is
  computed or seen (prompt Part 2). Episodes the operator cannot label are recorded and excluded, with the count.

## 3. Probes and metric (frozen now; no refit on the fresh rollouts)
- **Rest probe:** StandardScaler + LogisticRegression(C = 1.0, class_weight = balanced, max_iter = 5000) — the
  estimator of `scripts/analyze_controls.py` `loso` — fit once on **all 69** old rollouts at the frame those rollouts
  were read at (`judge_frame` of `projects/surveyor/results/judge.csv`), for each of: W1 latent, frozen DINOv2-S
  (`scripts/extract_controls.py` w2p), commanded joints, policy identity (one-hot checkpoint). On a fresh episode the
  same frame rule is applied and the frozen weights score it unchanged.
- **Grasp-anchor probe:** the anchor is `run_w1c_offline.c1_anchor` (first gripper close − 15 frames); the current
  feature is W1's latent at the anchor, the imagined feature is `imagine_at` (K = 30 over the recorded actions). One
  StandardScaler + LogisticRegression(C = 1.0, class_weight = balanced, max_iter = 10000) — the estimator of
  `probe_cv` — fit once on all scored old rollouts per feature (current, imagined; W2′ and joints current as
  controls), applied unchanged to the fresh ones (prompt: "probe weights fit on the 69 old rollouts and applied
  unchanged to the new ones — no refit").
- **Frozen weights:** `scripts/freeze_probes.py` writes `expected/frozen_probes.npz` and its sha256 into
  `expected/frozen_probes.json`; that commit is made **before the first fresh rollout** (owner 2026-11-02: "freeze
  the probes and commit before any arm trial"). The fit uses the old rollouts only, so it is fully determined now.
- **Colour judge:** the hand-built rule, no learning, scored as on the 69.
- **Metric:** AUROC of each frozen probe's score against the operator label on the fresh episodes; 95 % episode
  bootstrap, 2,000 resamples, seed 0, percentile, episodes resampled jointly across features for paired differences.

## 4. Baselines
Frozen DINOv2-S, commanded joints, policy identity (the identity null: does the checkpoint alone predict the outcome)
and the colour judge, all on the same fresh episodes (same-set comparisons only).

## 5. Bars (verbatim from the prompt, Part 2)
| # | bar | pass | fail | not proven |
|---|---|---|---|---|
| B1 | rest AUROC ≥ 0.85 for our model | AUROC ≥ 0.85 | AUROC < 0.85 | not given (O4) |
| B2 | ≥ the frozen-DINOv2 control by ≥ 0.05 (episode bootstrap CI excluding 0) | difference ≥ 0.05 and its 95 % CI excludes 0 | otherwise | not given (O4) |
| B3 | anchor: current ≥ imagined + 0.05 | current − imagined ≥ 0.05 | current − imagined < 0.05 | not given (O4) |

Source of every bar: prompt Part 2. The split is **confirmed** only if B1, B2 and B3 all pass; then the paper's
"exploratory" qualifiers on the split become "confirmed on <n> fresh rollouts (bar committed `<this commit>`)".
Whatever the verdict, the numbers go into a "Confirmation" paragraph of §4 and a row of the controls table.

**Read-back clarifications (wording only).** B2's 0.05 is read on the point difference, and the CI clause is read
on the same paired bootstrap. The anchor probe's fit set is the old rollouts scored at the anchor
(`expected/controls.json` `w1c1/scored`); an episode whose gripper never closes has no anchor, is not scored at the
anchor (as in `run_w1c_offline.py`) and is counted. Bootstrap resampling, single-class resamples and AUROC ties follow
`boot` in `scripts/analyze_controls.py` unchanged; LogisticRegression uses its default deterministic solver.

## 6. Checks before numbers
1. `frozen_probes.npz` sha256 matches `frozen_probes.json` at the commit that predates the first fresh rollout.
2. Two fresh episodes end to end: shapes of every feature, the anchor frame by hand, labels present before scores.
3. The frozen probes re-scored on the 69 old rollouts reproduce an in-sample AUROC (sanity only, not a result).

## 7. Deviations
D1 (2026-11-07, before any fresh rollout; `expected/confirm_rows.jsonl` does not exist): the owner delegated the open items O1-O6 to the agent's recommendation. The answers are written beside each item below. Only O4 adds a rule (the "not proven" verdict on fewer than 8 failures); it can only turn a pass into not proven, and no bar value in §5 changes.

## 8. Exploratory
Everything not in §5: per-failure-mode recall, seed sensitivity, any other frame or feature.

## Open items (wording, for the owner; no threshold changed)
- O1: the prompt names the rest feature "the final-frame latent"; the 69 were read at `judge_frame` (the colour
  **Answer (owner-delegated to agent recommendation 2026-11-07):** keep `judge_frame`. Reason: the frozen probe was fit at that frame on the 69, and a different frame type on the fresh episodes would be a refit or a distribution shift the frozen weights never saw.
  judge's rest frame). This protocol uses the same `judge_frame` rule so the frozen probe sees the same frame type.
- O2: the prompt's "the LOSO linear probe" is read as the LOSO probe's estimator fit once on all 69 and frozen (a LOSO
  **Answer (owner-delegated to agent recommendation 2026-11-07):** keep the reading (estimator of LOSO, fit once on all 69, frozen). Reason: a LOSO refit on the fresh rollouts is a refit, which the prompt rules out for the anchor probe and the owner's "freeze the probes" instruction rules out for all.
  refit on the fresh rollouts would be a refit, which the prompt rules out for the anchor probe).
- O3: B3 has no CI clause in the prompt; the CI is reported, the bar is read on the point difference.
  **Answer (owner-delegated to agent recommendation 2026-11-07):** keep: CI reported, B3 read on the point difference, as B3's wording in the prompt has no CI clause. Reason: adding a CI clause to B3 would make its bar stricter than the one committed first; B2 already carries its CI clause.
- O4: the prompt gives no "not proven" rule (e.g. for fewer than 8 failures at the 40-rollout stop); until the owner
  **Answer (owner-delegated to agent recommendation 2026-11-07):** if fewer than 8 failures after the stop at 40, a bar that would pass is reported **not proven** (the failure count beside it), never pass; a bar that fails on the point estimate stays fail. Reason: the prompt's target of at least 8 failures is the stated minimum for the test to say anything on failures, and the conservative reading never confirms the split on fewer; it cannot turn a fail into a pass.
  gives one, the verdict is pass / fail as stated and the failure count is reported beside it.
- O5: whether episodes excluded for no operator label count toward the 30 / 40 is not given; until the owner says,
  **Answer (owner-delegated to agent recommendation 2026-11-07):** keep: episodes excluded for no operator label do not count toward the 30 / 40, and their number is reported. Reason: only labelled episodes enter the AUROC, so only they can make the set fresh and sized as the bar assumes.
  they do not count, and their number is reported.
- O6: "same SmolVLA checkpoints" sets no per-checkpoint balance or order; the operator's run order is recorded.
  **Answer (owner-delegated to agent recommendation 2026-11-07):** keep: no balance or order constraint; the operator's run order and the per-checkpoint counts are recorded and reported. Reason: the prompt sets none, and adding one now would be a new rule; the policy-identity null already shows whether a checkpoint imbalance carries the result.
