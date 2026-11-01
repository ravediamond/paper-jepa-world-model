# W1's training set vs the 69 rollouts (owner check, 2026-11-01 evening)

| what | value | source |
|---|---|---|
| W1 checkpoint | `ravediamond/lewm-w2-so101-bowls` `w1/weights_final.pt` @ `d0336db6` | `data/FACTS.json` → `w1_checkpoint` (source: EXPERIMENTS.md row G0) |
| W1 trained | 2026-10-11 (G0, 100 epochs) | `journal/EXPERIMENTS.md:63` |
| training data | `ravediamond/so101_bowls_all_trimmed` @ `4e5c6fda8dc5…`, 157 train / 25 val episodes | `projects/vla-jepa-world-model/wm_train/split.json` (`dataset`, `revision`, `n_train`, `n_val`); `prepare_data.py:34` `REPO` |
| what that dataset is | "Bowls teleop dataset", 182 episodes (90 / 92 per task), idle starts trimmed | `journal/EXPERIMENTS.md:171` |
| the 69 rollouts | `ravediamond/so101-bowls-policy-rollouts` @ `bad1c8e`, a separate dataset, published 2026-10-14 (after G0) | `data/FACTS.json` → `rollouts`; `journal/EXPERIMENTS.md:173` |

**Verdict: W1 is teleop-only.** The rollouts are not in W1's training data (different repo, published three days after
the checkpoint): they are out-of-sample for the encoder. The earlier wording "W1 saw these 69 rollouts as unlabelled
video in training" (FEASIBILITY.md G10 row, OUTLINE_4PAGE.md, gaps/README.md) had no source and is withdrawn.

What stays: the readouts in `controls.json` are linear probes fit on the same 69 rollouts with leave-one-session-out
cross-validation and operator labels, and no bar was committed before them — they remain **exploratory**.
