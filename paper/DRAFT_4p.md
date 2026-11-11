# A small JEPA world model on a low-cost arm: the encoder reads the outcome, the predictor adds nothing to it

*Ravindu Somawansa, Shisso Robotics, Paris. Draft, not submitted.*

## Abstract

We trained a small JEPA world model in the LeWorldModel recipe for about $3 of rented GPU time. The training data are teleoperated demonstrations of one tabletop task on an SO-101 arm. We then re-tested on this model the uses of such models that recent papers propose. Each use had a threshold fixed before its test. We tested the encoder on policy rollouts the model never saw. A linear probe on its last-frame latent separates success from failure about as well as a hand-built colour judge. It does better than a frozen DINOv2 encoder and better than the commanded joints. The predictor adds nothing to this. At the grasp, the imagined latent reads the outcome at 0.70 and the current latent at 0.82. Every use that needs the predictor to roll forward misses its threshold. It still does when we train the same recipe on 45 to 2,909 episodes, which adds a second dataset and then another lab's rollouts. The controls are exploratory and come from one rig and one model per size.

## Introduction

![What the world model reads and what it forecasts. The encoder's last-frame latent separates success from failure on rollouts the model never saw about as well as a hand-built colour judge, and it keeps doing so when we retrain on 45 to 2,909 episodes. The predictor misses its threshold at the original size and at every other size: the commanded actions rank the executed chunk better than the imagined latents do.](fig/encoder_predictor.pdf){full}

LeWorldModel (LeWM) [1] trains a world model in two parts. The encoder maps camera frames to a latent vector. The predictor takes the current latent and a sequence of actions and imagines the latents that follow. Both are trained together from pixels with a joint-embedding predictive (JEPA) loss. Most of the uses that later papers propose need the predictor. LeWM itself ranks a policy's candidate action chunks by their imagined outcome. ContactGuard [2] reads the outcome of a grasp before it happens, HWM [3] and H-JEPA [4] plan over several time scales, and other papers flag a failure when the observed latent drifts from the predicted one. Each of these papers reports that its use worked on its own robot and data.

We had one such model, trained on our own arm. We ran every use we could find on it, 20 tests in all. 13 of them re-test a claim of a paper, and 7 are uses we designed ourselves on the same model. One test is not a use of our model: it times the pretrained RoboJEPA [5] and V-JEPA 2.1 [6] encoders on our on-board computer. 14 of the 20 tests fail, 4 hold and 2 are not proven. Table 1 groups them by what they ask of the model.

The first version of this paper argued something completely different. We read the failures as a sign that the latent mostly encodes the arm's pose. Then a frozen image encoder or the commanded joints would do at least as well. We ran these controls on 69 policy rollouts. The encoder read how the episode went better than either control. We modified the claim around the predictor.

We come back several times to one moment, the grasp anchor. It is 15 frames before the gripper first closes. There the current latent reads the outcome at 0.82 and the imagined latent at 0.70.

The threshold of each test was fixed before its result; for the two-level progress test it was written down together with the result. The controls that split the encoder from the predictor were added after the data and had no threshold, so everything this paper says about that split is exploratory.

## Setup

The robot is one SO-101 arm [7] driven through LeRobot [8], with a top and a wrist camera; it runs on position control, so the arm goes close to where it is commanded. The task is to put a bowl into another bowl. We trained our model on `ravediamond/so101_bowls_all_trimmed`, 182 teleoperated episodes of about 11 s each (34 minutes in all), split by grid layout into 157 training and 25 validation episodes. The evaluation rollouts come from a separate dataset, `ravediamond/so101-bowls-policy-rollouts`. They are SmolVLA [9] runs recorded after our model was trained: 69 episodes, 44 successes and 25 failures over 13 sessions. Most failure monitors use the main set of 50 episodes (42 successes, 8 failures), which leaves out the episodes of one policy family. The author ran the arm and labelled every outcome, and was not blind to the policy.

These data limit what the tests can show. The task is short and has no long horizon, no memory and no multi-step structure. Everything was recorded in one place, with one arm and one task family. The main set has 8 failures, so one episode can move a monitor's verdict. Every test ran offline on recorded episodes, never in closed loop on the arm.

Our model follows the LeWM recipe [1], with a multi-view ViT encoder, an action-conditioned predictor and the SIGReg regulariser. We compare it with four cheaper signals. The frozen DINOv2-S control is a frozen DINOv2 small encoder [10], mean-pooled over the top and wrist views as in DINO-WM [11]. The joints control is the commanded joint vector. The colour judge is a hand-built rule on the final frame with no learning. Policy identity is the checkpoint that ran the episode.

One timing test loaded a random encoder by mistake, because a library version silently dropped 192 encoder tensors; we voided its latents, and every other test ran the trained model (Appendix B).

## What we tested

Table 1 sorts the 20 tests by what they ask of the model, and Table 5 in the appendix gives each one with its threshold and result. Each group has a cheap competitor that needs no world model, run on the same episodes. Where we compared a method with its competitor, a paired bootstrap over episodes (2,000 resamples, 95 % interval) says whether the difference could be luck: an interval entirely below zero means the method lost.

Table 1: The 20 tests grouped by what they ask of the model, with the cheap competitor each group is compared with and the verdicts (held / failed / not proven). Every test is listed in Table 5.

| what the test asks of the model | tests | cheap competitor | held | failed | not proven |
|---|---|---|---|---|---|
| read the outcome of an episode from the latent | 3 | a hand-built colour judge; the commanded joints; a frozen DINOv2-S encoder; the current latent instead of the imagined one | 1 | 1 | 1 |
| read the arm's pose, the task's progress or stage | 7 | the measured or commanded joints; a frozen DINOv2-S encoder | 2 | 4 | 1 |
| predict the next latents and rank or plan action chunks | 5 | copy-last (the next latent is the current one); the commanded actions alone | 1 | 4 | 0 |
| flag a failing episode from the predictor's surprise or risk | 4 | the risk score with shuffled actions (ReAP); the other monitors have only their threshold | 0 | 4 | 0 |
| pretrained RoboJEPA and V-JEPA 2.1: run the encoder on the on-board computer, predict ahead | 1 | the 200 ms real-time budget; copy-last | 0 | 1 | 0 |

The cheap competitor is strong on this rig. The arm runs on position control, so the commanded joints already say where it will end up. The task is short, and little else in the scene moves. A learned latent has little room to add to what the commands carry. On a long task, or with a compliant arm or moving objects, we would expect this competitor to be weaker.

Each failure is also labelled with whose method failed (Table 5, the same labels as our re-test of robot-learning papers on this rig). The authors' method, or the authors' method with only what our rig forced changed, counts against the paper's claim on this rig. Our variant of a method, a test outside the paper's regime, and a use of our own do not.

## What the encoder knows

Among the encoder uses, reading the outcome works. The outcome readout at rest passes. At the grasp anchor the current latent reads the grasp at 0.82. Reading progress does not work. Four uses ask the latent for progress or task stage, and all four fail. They are progress monotonicity, two-level hierarchical progress [4], the latent stage tracker and SURVEYOR's latent-distance scale [12]. We also ran the pose probes of LeJEPA [13]. A linear readout recovers about half of the joint variance from our encoder (R² 0.565). It recovers more from the frozen DINOv2-S control (0.867).

We tested whether the encoder carries anything that a frozen encoder or the joints do not. Table 2 sets our model beside the four other signals on the same 69 rollouts. Every probe is the same leave-one-session-out logistic regression, fit against the author's labels (Appendix C). At rest the interval is a 95 % episode-bootstrap CI. At the anchor it is a bootstrap of the mean over 5 seeds of the out-of-fold AUROC, with episodes resampled jointly across the seeds (2,000 resamples, percentile).

Table 2: Outcome AUROC of each signal on the same 69 rollouts (exploratory: controls chosen after the data). No judge applies at the anchor.

| feature | at rest | grasp anchor (65 scored, 21 fail) |
|---|---|---|
| our world model | 0.9545 [0.9023, 0.9908] | 0.8199 [0.7141, 0.9041] |
| colour judge (no learning) | 0.9432 [0.8919, 0.9878] | — |
| frozen DINOv2-S | 0.8655 [0.7767, 0.9405] | 0.7067 [0.5816, 0.8269] |
| commanded joints | 0.7573 [0.6084, 0.8880] | 0.5305 [0.3709, 0.6855] |
| policy identity | 0.5882 [0.4084, 0.7515] | — |

![Table 2 as a dot plot. Left: at rest, with the 95 % CI. Right: at the grasp anchor, with the seed range (not a CI).](fig/table2.pdf)

> **Finding.** At rest our model is level with the colour judge. The two CIs overlap. Our model is above the frozen encoder and the joints. At the grasp anchor no judge applies, and our model is above both controls. The pose probes hold the joints better than our encoder does. They hold the outcome worse. On these rollouts the encoder has outcome information that the pose alone does not give.

The anchor column depends on how the five seeds are combined. If we average the predictions before scoring, our model gets 0.8377, frozen DINOv2-S 0.7175 and the joints 0.5346. With seed 0 alone they get 0.7554, 0.7024 and 0.5574. The gap to the frozen encoder is smaller with seed 0. The order is the same.

On SURVEYOR's scale check the controls do better than our model. Frozen DINOv2-S has τ 0.3601 and the joints τ 0.0265, and both pass. Our model has τ 1.2488 and fails. SURVEYOR reports 0.20. The task and the rollouts are the same for all three features. We take the failure as a property of our latent space.

The first scoring of the outcome readout reports 1.000. That number was scored against the colour judge's labels on 87 rollouts. Scored against the author's labels on the 69 rollouts, the same readout gives 0.9545.

## Where the predictor fails

The predictor does predict its own latent. At K = 30 its open-loop error is 0.31, and copying the last latent gives 0.89. The uses of this prediction fail. In chunk ranking [1] the model puts the executed chunk first in 0.60 of anchors, against a threshold of 0.8. The commanded actions alone reach 0.68–0.90. The surprise monitor catches 2/8 at 2/42 failures. The hierarchical planner [3] and the risk head [14] also miss their thresholds. So do the conformal surprise band, re-ranking and the calibrated reflex. Table 5 gives the margins.

Neither result is luck. Against copy-last the paired difference in latent error is +0.58 [+0.51, +0.66] in favour of the model. In chunk ranking the model's top-1 minus the commanded actions' is −0.37 [−0.42, −0.32] at d = 10 and −0.23 [−0.29, −0.18] at d = 5 over 25 validation episodes: the interval is below zero at every distance we scored. Predicting well is not ranking well. The pretrained RoboJEPA predictor also beats copy-last, by less than its claim asks: its 9-step latent error is 11.8 % below copy-last (interval 10.8 % to 12.9 %, 25 episodes), short of the 15 % its claim asks, and its encoder takes 526 ms for the two views on our on-board computer against a 200 ms budget.

ContactGuard [2] finds that probes on the imagined latent beat probes on the current latent for grasp tasks (AUROC 0.98 vs 0.84 on its cup task). We expected the same at our grasp anchor, and the threshold fixed before the test asked the imagined latent for ≥ 0.85, ≥ +0.05 over the current one. We found the reverse. With the same encoder and the same anchor, the current latent reads the outcome at 0.82 and the imagined latent at 0.70. The paired difference is −0.12 [−0.24, −0.01] over 65 rollouts, so the loss is not luck. The row fails. ContactGuard used one policy and three cameras; we used mixed policies and two.

Two monitors flag failures from the predictor. FAIL-Detect [15] and SAFE [16] set a conformal band on a failure score from successful episodes; we used our model's surprise as the score, not their scores, so this is our variant. Our first band was built wrong, calibrated on the same episodes it was fitted on. With the band built as the papers build it (11 + 11 successes to build the band, medians over 20 random splits), 9.1 % of held-out successes are flagged at α = 5 % (not 22.7 %) and 24 % of failures are caught (not 46 %); both still miss. ReAP [14] learns a risk head on the latent and the planned actions. Our first version on our latent caught 3/8 at 5/42; rebuilt from the paper with its Stage 1 detector and Stage 2 predictor (no code is released, one seed), it caught 2 of 8 at 1 of 42 false interventions. Both miss 5 of 8. The risk head reads the true actions better than shuffled ones (0.79 against 0.71 AUROC), but the interval of the difference, +0.08 [−0.05, +0.20], contains zero, so we cannot tell.

We also ran the distillation probe on our model's inverse-dynamics term, pooled over the 25 teleoperation validation episodes. The trained model scores 0.7476. Its random initialisation scores 0.7974. The difference is −0.0499, with a 95 % paired episode-bootstrap CI of [−0.7403, +0.6559]. The term depends on the machine. The same untrained network scored 1.4383 on one machine and 0.7974 and 0.8042 on two others. It is sensitive to the BLAS library and to ridge conditioning.

The surprise monitor has a failure-class AUROC of 0.775, below its sanity bar. We checked it for a loading or scoring bug. The same scorer reproduces the open-loop corruption ratios at K = 30. Cross-episode gives 3.428, against 3.341 in the original run. Constant-first gives 2.558, against 2.543. We found no bug.

We ran an action-shuffle check on the predictor of VLA-JEPA at 5 anchors (Table 3). Action tokens from another episode differ from the true ones by 21–28 % in relative L2. They move the loss by about 0.05 %. Zeroing the tokens moves it by about 1.05 % on average. The forward pass is deterministic. The predictor barely uses its actions. Its failure-class AUROC is 0.501, below the same sanity check, and we leave its table unread under the frozen bar.

Table 3: VLA-JEPA world-model loss at 5 anchors with true, shuffled (Δ shuffled) and zeroed action tokens (exploratory spot check). The token change is the relative L2 of the shuffled tokens.

| anchor (episode / frame) | true loss | Δ shuffled (%) | token change (%) | Δ zeroed (%) |
|---|---|---|---|---|
| 8 / 402 | 1.3031 | +0.014 | 21.1 | +2.22 |
| 10 / 227 | 1.4641 | +0.088 | 27.6 | +0.50 |
| 23 / 327 | 1.4215 | +0.036 | 21.1 | +0.87 |
| 59 / 302 | 1.4637 | +0.030 | 25.5 | +1.28 |
| 65 / 302 | 1.5236 | −0.070 | 22.0 | −0.38 |

The outcome-readout gate has one pass that uses the imagined latent. In its anticipation step the imagined latent beat policy identity by +0.179. We set that bar after the result, so it is exploratory. We then reran the idea twice with the threshold committed first. One run was the grasp probe and the other the calibrated surprise reflex. Both fail.

## Does more data change this?

All of this comes from one model trained on 157 episodes. We trained the same recipe at three other sizes, for the same 81,068 gradient steps as the original model, and ran the same probes on each checkpoint (Table 4). The smallest model sees 45 of the 157 episodes. The original model is the second row and was not retrained. The third adds 653 episodes of a second cubes-and-bowls dataset of ours at the same frame rate. The fourth adds the 2,099 policy rollouts of ArmnetBench, which come from another lab with other cameras and which we kept at their native 20 Hz, so the step to the last row changes the data and the setup together. The probes, thresholds and decision rule were fixed before any of these runs.

Table 4: The probe battery on one model per training-set size. Chunk ranking is the top-1 of 8 candidates at the longest horizon (d = 30) with its gap to the commanded-actions ranker (0.873 top-1 on the same anchors) and the 95 % episode-bootstrap CI of the paired gap. The beyond-state columns come from a ridge probe that predicts the commanded action from the latent now and 5 frames later: the first number is its residual R² above the latent now alone or paired with another episode's latent, the second is what the two latents add to the measured joints. The surprise monitor column is the failures caught of 8 at 2 of 42 false interventions. The rest readout is scored on the 69 rollouts against the author's labels.

| training episodes (frames) | joints R² | chunk ranking, d = 30 | beyond-state gain / over joints | surprise monitor | rest readout AUROC [CI] |
|---|---|---|---|---|---|
| 45 (15,032) | 0.556 | 0.403, gap -0.471 [-0.539, -0.405] | 1.33 / 0.006 | 1 of 8 | 0.931 [0.862, 0.982] |
| 157 (53,015) | 0.565 | 0.605, gap -0.269 [-0.334, -0.204] | 0.75 / 0.007 | 2 of 8 | 0.955 [0.902, 0.991] |
| 810 (275,015) | 0.565 | 0.273, gap -0.600 [-0.652, -0.545] | 0.63 / 0.007 | 1 of 8 | 0.956 [0.899, 0.996] |
| 2,909 (1,282,161) | 0.573 | 0.222, gap -0.651 [-0.696, -0.602] | 1.17 / 0.009 | 0 of 8 | 0.897 [0.813, 0.964] |

At d = 30 the commanded actions rank the executed chunk first at 0.873. Our model is below them at every size by a gap whose CI excludes zero. Its best cell was 0.422 with 45 episodes, 0.605 with 157, then 0.279 and 0.230 after we added the cubes dataset and ArmnetBench. The 0.8 threshold is out of reach at all four sizes. We have not tested why the larger sets did worse. The added episodes are other tasks and another rig, and the ranking is scored on the same 25 bowls episodes each time, so the added data may take capacity from the task we score. The surprise monitor caught 1, 2, 1 and 0 of 8 failures. It needed 5.

Neither encoder probe moves with the data. The joints R² is 0.556 at the smallest size and 0.573 at the largest. The rest readout is level with the colour judge (0.943) in every row, since the CI of the paired difference includes zero, although the largest model has the lowest AUROC, 0.897. For the beyond-state probe, the gain of the latent 5 frames later over the latent now alone is at least 0.62, but what it adds to the measured joints is 0.006 to 0.009 against a threshold of 0.02, and the untrained encoder gains 1.25 from the same extra latent, more than half of any trained one's gain. The verdict is not proven at every size.

This is one run per size with one seed. The decision rule asks for two seeds at the smallest and the largest size, and for a predictor trained on frozen DINOv2-S features at every size as the floor of the surprise monitor. We ran neither. Read it as four runs in which no use of the predictor improves with data, not as a statement about the model class.

## Related work

LeWorldModel [1] trains the recipe we use end to end from pixels and plans with it in Push-T and in simulation. We tested its two planning uses on our arm, and both fail. LeJEPA's identifiability analysis [13] predicts that SIGReg latents identify the state up to anisotropy. Our pose probes support this in part. DINO-WM [11] plans on frozen DINOv2 [10] patches, and our frozen control uses the same design. We timed V-JEPA 2.1 [6] and RoboJEPA [5] on our on-board computer, and both are too slow. ContactGuard [2] finds imagined-latent probes better than current-latent ones on grasp tasks. On our arm we find the reverse. We also re-tested ReAP's action-conditioned risk [14], the conformal bands of FAIL-Detect [15] and SAFE [16] on our surprise score, the hierarchical planners of HWM [3] and H-JEPA [4], and SURVEYOR's latent-distance scale [12]. All of them miss here, at the original data scale. SaiVLA-JEPA [17] reports that a training-only future-latent head adds +0.03 points to control on 10,030 paired episodes. HIJP [18] finds LeWM below goal-conditioned behaviour cloning in simulation.

## Limitations

We chose the probes for the encoder and predictor split after the data, so the split is exploratory. They are fitted on 69 rollouts that one operator, the author, labelled. The model never saw these rollouts, and every probe is cross-validated by session. To confirm or reject the split we need a fresh set of rollouts scored under a threshold committed first. That takes a day on the arm. The main results come from one model at one data scale. The data-scale runs add three sizes, one run and one seed each. A second seed at the smallest and the largest size, the frozen-DINOv2-S and random-encoder predictors at each size, the 90-episode size and a successes-only variant of the largest were not run. Three controls were not run. The surprise monitor and the imagined arm of the grasp probe would each need a predictor trained on frozen DINOv2-S features or on the joints. Progress monotonicity with the controls would need a reimplementation. The pass of the outcome readout is retrospective: its anticipation bar was set after the result.

## Conclusion

> **Conclusion.** We ran 20 uses on a world model that cost about $3 to train. The encoder can tell a success from a failure after the fact. The predictor does not add to that, and training on up to 2,909 episodes did not change it. For ranking chunks the commanded actions were better than the predictor, and they need no model.

## Appendix A: every test

Table 5: Every test run on our world model (with the frozen-DINOv2 and RoboJEPA tests beside it), what it takes from the model, whose method it is, the threshold, the result and the verdict. Whose method: the authors', the authors' fitted to our rig (only what the rig forced was changed), our variant of it, outside the paper's regime, or a use of our own with no paper claim behind it. Bar, result and verdict are quoted from the stored results; the conformal band and the ReAP row carry the corrected and the rebuilt result.

| test | re-tests | takes | whose method | bar | result | verdict | code |
|---|---|---|---|---|---|---|---|
| open-loop latent prediction | [1] | predictor | — | beats copy-last; ratios ≥ 1.3 | open loop K = 30: 0.31 vs copy-last 0.89; timing shifts 1.99 / 1.58; rank 24–30 | pass | G0 |
| imagined- vs current-latent outcome probe (first run) | [2] | both | authors', fitted to our rig | imagined beats current | imagined 0.72, current 0.68; policy identity 0.76 | not proven (policy confound) | C1 |
| action-chunk ranking | [1] | predictor | authors' | top-1 ≥ 0.8 | top-1 best 0.60; action-only 0.68–0.90, regret ≈ 0 (52) | fail | G1b |
| arm-pose linear probe | [1, 11] | encoder | — | > 0.57 | arm-pose R² ours 0.57, **DINOv2-S 0.92** | holds for DINOv2-S | G1 |
| chunk ranking with a frozen-DINOv2 world model | [11] | predictor | authors' | ≥ 0.8 | DINOv2-S top-1 0.55–0.61 | fail | G1-W2 |
| progress monotonicity and done detection | this project | encoder | our own use | ρ ≤ −0.8; AUROC ≥ 0.95 | monotonicity −0.32 to −0.50; done AUROC ours 0.97 (easy contrast); 32 % false stalls | fail (done detection passes as a sanity check) | G6a |
| outcome readout at rest; imagined-latent anticipation | [2] | both | our own use | ≥ 0.90; +0.10 | readout at rest AUROC ours 1.000, DINOv2-S 0.976; anticipation +0.179 over policy identity | holds (bar set after the result) | G10 |
| two-level hierarchical progress | [4] | encoder | outside its regime | ≤ −0.80; ≤ 10 % | H-JEPA level 2 ρ −0.727; false stalls 52 % | fail | G6b |
| surprise monitor | this project | predictor | our own use | ≥ 5/8 at ≤ 1/20 | L2 surprise 2/8 at 2/42 (AUROC 0.86); drift 0/8; live reflex 42/42 successes flagged | fail | W1-A |
| latent stage tracker | this project | encoder | our own use | ≤ 2 %, ≤ 10 % | 22.1 % premature, 31.1 % missed | fail | W1-B |
| calibrated surprise reflex | this project | predictor | our own use | recall ≥ 50 % | 3/25 failures at 1/44 | fail | G11-B-cal |
| imagined-latent grasp probe | [2] | both | authors', fitted to our rig | ≥ 0.85, ≥ +0.05 | imagined 0.70 < current 0.82 | fail | W1-C1 |
| conformal surprise band | [15, 16] | predictor | our variant | ≤ 1/20, ≥ 50 % | papers' band, medians over 20 splits: successes 9.1 % flagged, failures 24 % (our first band, built wrong: 22.7 % and 46 %) | fail | W1-C2 |
| re-ranking chunk candidates | [1] | predictor | our own use | ≤ 0.10 | regret 0.2631 vs ours 0.2630, action-only −0.0075 | fail | W1-R |
| action-conditioned risk head | [14] | both | authors', fitted to our rig | ≥ 5/8 at ≤ 1/20 | first version 0.93 withdrawn (time leak); second version 3/8 at 5/42, AUROC 0.79 vs shuffled 0.71; Stage 1 + Stage 2 rebuilt from the paper: 2/8 at 1/42 (one seed) | fail | REAP |
| inverse dynamics from a latent pair | this project | encoder | our own use | random-init < half ours | latent pair +0.748 over one frame, but random-init ours 1.254 vs ours 1.180; beyond state +0.007 | not proven | §30 |
| linear identifiability of the latent | [13] | encoder | — | R² < 0.80; ≥ 0.20 | ours mean R² 0.565, anisotropy 0.639; DINOv2-S 0.867 / 0.301 | holds | LeJEPA |
| latent-distance scale for plan consumption | [12] | encoder | authors' (scale); our variant (monitor) | τ ≤ 0.40; ≥ 5/8 at ≤ 1/20 | τ 1.249 (paper 0.20); monitor 3/8 at 6/42 | fail | SURVEYOR |
| hierarchical vs flat prediction horizon | [3] | predictor | authors' | crossover in [1, 2] s | no crossover: high level 0.76 vs ours 0.56 at 2 s | fail | HWM |
| on-device latency of RoboJEPA / V-JEPA 2.1 | [5, 6] | both | authors' (prediction); our own use (timing) | ≤ 200 ms; ≥ 15 % | encoder 526 ms (2 views); ≤ copy-last + 12 % | fail | RoboJEPA |

## Appendix B: the loading bug

In October we found a loading bug. A timing test on the on-board computer loaded our checkpoint with `strict=False`. The image had transformers 5.5.4, and under that version 192 encoder tensors did not match. They were dropped without an error. The test timed a random encoder, and we voided its latents. We then traced how every other use loaded the model. 26 ran the trained model and 1 ran a random one, the timing test. The world-model block of VLA-JEPA, which one diagnostic above uses, loads all 351 of its keys.

## Appendix C: the probe

Every outcome probe in Table 2 is this function (`repro/gaps/scripts/analyze_controls.py`):

```python
def loso(X, y, groups):
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        sc = StandardScaler().fit(X[~te])
        m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X[~te]), y[~te])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p
```

Every number in this paper is generated by `paper/make_paper.py` from `repro/*/expected/`; its source is in `paper/numbers.json`.

## References

[1] Maes, Lucas, Le Lidec, Quentin, Scieur, Damien, LeCun, Yann, Balestriero, Randall. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv preprint arXiv:2603.19312, 2026.
[2] Zheng, Gehan, Johnson-Roberson, Matthew, Zhi, Weiming. ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models. arXiv preprint arXiv:2608.13438, 2026.
[3] Zhang, Wancong, Terver, Basile, Zholus, Artem, Chitnis, Soham, Sutaria, Harsh, Assran, Mido, Balestriero, Randall, Bar, Amir, LeCun, Yann, Ballas, Nicolas. Hierarchical Planning with Latent World Models. arXiv preprint arXiv:2604.03208, 2026.
[4] Zhang, Wancong, Terver, Basile, Rabbat, Michael, LeCun, Yann, Balestriero, Randall. H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning. arXiv preprint arXiv:2610.06805, 2026.
[5] Zholus, Artem, Beltran-Velez, Nicolas, Yuan, Jianhao, Chandar, Sarath, Nagarajan, Tushar, Severo, Daniel, Sinha, Koustuv, Drozdzal, Michal, Romero Soriano, Adriana, Bohg, Jeannette, Ballas, Nicolas, Assran, Mahmoud. RoboJEPA: Scaling Robotic Latent World Models. arXiv preprint arXiv:2610.10515, 2026.
[6] Mur-Labadia, Lorenzo, Muckley, Matthew, Bar, Amir, Assran, Mido, Sinha, Koustuv, Rabbat, Mike, LeCun, Yann, Ballas, Nicolas, Bardes, Adrien. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. arXiv preprint arXiv:2603.14482, 2026.
[7] TheRobotStudio, Hugging Face. SO-ARM100 / SO-101: Standard Open Arm. https://github.com/TheRobotStudio/SO-ARM100, 2025.
[8] Cadene, Remi, Alibert, Simon, Soare, Alexander, others. LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch. https://github.com/huggingface/lerobot, 2024.
[9] Shukor, Mustafa, Aubakirova, Dana, Capuano, Francesco, Kooijmans, Pepijn, Palma, Steven, Zouitine, Adil, Aractingi, Michel, Pascal, Caroline, Russi, Martino, Marafioti, Andres, Alibert, Simon, Cord, Matthieu, Wolf, Thomas, Cadene, Remi. SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv preprint arXiv:2506.01844, 2025.
[10] Oquab, Maxime, Darcet, Timothée, Moutakanni, Théo, others. DINOv2: Learning Robust Visual Features without Supervision. arXiv preprint arXiv:2304.07193, 2023.
[11] Zhou, Gaoyue, Pan, Hengkai, LeCun, Yann, Pinto, Lerrel. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. arXiv preprint arXiv:2411.04983, 2024.
[12] Anonymous. SURVEYOR: Certified Speculative Plan Consumption for Latent World-Model Planning. NeurIPS 2026 submission; no arXiv identifier known, 2026.
[13] Klindt, David, LeCun, Yann, Balestriero, Randall. When Does LeJEPA Learn a World Model?. arXiv preprint arXiv:2605.26379, 2026.
[14] Liu, Jingyuan, Qing, Yunpeng, Wei, Changxu, Hou, Zhi. ReAP: From Visual Failure Recognition to Action-Conditioned Risk Prediction for Runtime Failure Monitoring. arXiv preprint arXiv:2610.16309, 2026.
[15] Xu, Chen, Nguyen, Tony Khuong, Dixon, Emma, Rodriguez, Christopher, Miller, Patrick, Lee, Robert, Shah, Paarth, Ambrus, Rares, Nishimura, Haruki, Itkina, Masha. Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies. arXiv preprint arXiv:2503.08558, 2025.
[16] Gu, Qiao, Ju, Yuanliang, Sun, Shengxiang, Gilitschenski, Igor, Nishimura, Haruki, Itkina, Masha, Shkurti, Florian. SAFE: Multitask Failure Detection for Vision-Language-Action Models. arXiv preprint arXiv:2506.09937, 2025.
[17] Xiao, Rufeng, Xiao, Ruihao, Chen, Wei, Sun, Xinhai. SaiVLA-JEPA: Evaluating Future Latent Prediction in JEPA-Conditioned VLA Control. arXiv preprint arXiv:2610.38508, 2026.
[18] Du, Guanchen, Du, Jinhao, Sun, Yanpeng, Zhang, Shan, Zhao, Na, Li, Zechao. Beyond Action Prediction: Learning Search-Free Planning in Latent Space. arXiv preprint arXiv:2610.23615, 2026.
