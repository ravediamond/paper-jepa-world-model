# Data-scale probes (WS-106)

The probe battery of the paper's data-scale section, run on one JEPA world model trained at four data scales with the same recipe and the same number of gradient steps (81,068, the count of the original model, W1).

| scale | training episodes | what is added | training frames |
|---|---|---|---|
| S1 | 45 | 45 of the 157 bowls training episodes (first 45 of a `numpy` permutation, seed 0) | 15,032 |
| S3 | 157 | the original model W1, not retrained | 53,015 |
| S4 | 810 | S3 plus 653 episodes of a second cubes-and-bowls dataset of ours, same rate (30 Hz) | 275,015 |
| S5a | 2,909 | S4 plus the 2,099 policy rollouts of ArmnetBench (another lab, other cameras, native 20 Hz) | 1,282,161 |

The 25 validation episodes, the 69 policy rollouts and their 50-episode subset are in no training list at any scale (`l0_check.py`). Each scale is one training run with one seed (3072).

## Files

- `PROTOCOL.md`, `AMENDMENT.md`, `FREEZE.md`, `CHECKS.md`: the frozen protocol and the owner's amendments, committed in the working repository before any number. `PROTOCOL.md` §7 (deviations) is empty. The `rate_rule: pending C4` field in the S4 and S5a split files predates Amendment 3, which fixed ArmnetBench at its native 20 Hz.
- `scripts/`: the scripts as they ran on the Orin (`probes.py`, `run_battery.sh`, `finish_scales.sh`, `run_all.sh`). They call the original scripts of the working repository and need the checkpoints, so they document the run and are not re-run by `make check`.
- `splits/`: the training lists of S1, S4 and S5a.
- `expected/<scale>/`: `p1/gate0.json` (P1), `p2.json` (P2), `p3/results_ridge.json` (P3), `p4/w1a_summary.json` (P4), `p5.json` (P5), `verdict.json` (the join), `training_end.json` (end record of the training log), `weights.sha256`. `expected/S3/c2.json` is check C2. `expected/curve.json` is the table of `probes.py table`.
- `l0_check.py`: recomputes each `verdict.json` from the five probe outputs and checks the statements of the paper.

## Probes (PROTOCOL.md §3, §5)

- P1 joints R² and anisotropy of the encoder (descriptive).
- P2 chunk ranking, top-1 of 8 candidates in the cells d = 5 to 30, against the action-only ranker (the floor) and the original 0.8 bar.
- P3 the inverse-dynamics ("beyond state") rule of the earlier gate, three conjuncts, with its random-init control.
- P4 the surprise monitor on the 50 rollouts: at least 5 of 8 failures at no more than 1 in 20 false interventions.
- P5 the rest readout of the 69 rollouts, against the colour judge.

## Not run

A second seed at S1 and S5a, S2 (90 episodes), S5b (successes only), S4 at fixed epochs, the frozen-DINOv2 predictor and the random-encoder predictor at each scale, so P4 has no W2′ floor and P1, P2 and P4 have no control lines. The paper says this in its limitations.

## Known issue found while writing

The first `probes.py table` counted the S5a training frames from all 2,499 ArmnetBench episodes (1,402,896 in total) and not from the 2,099 rollouts the model trained on. The split file and check C3 give 1,282,161, which `probes.py` now uses. No verdict used the frame count.
