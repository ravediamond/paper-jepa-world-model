# WS-106 checks before numbers (PROTOCOL.md §6)

## C1 — N and the recipe diff (2026-11-02)

- W1's run: Hub `ravediamond/lewm-so101-bowls` `runs/20261010T1941_train_55270119/` (the 100-epoch run named in
  `wm_train/README.md:171`). `out/multiview/metrics.jsonl` end record: `{"kind": "end", "step": 81068, "elapsed_s": 43339.3}`;
  last train record step 81060, epoch 99; epochs start every ~810 steps (epoch 1 at step 820, 99 at 80300).
- **N = 81,068.** W1 stopped on its `trainer.max_time 00:12:00:00` (config in the run dir and inside `w1/weights_final.pt`)
  ~32 steps before the end of epoch 99, with lr already 3.1e-11 (cosine end). The protocol's "≈ 81,100" estimate is
  replaced by the measured 81,068.
- Recipe diff, `wm_train/config/lewm_mv_so101.yaml` (today) vs W1's run `config.yaml`: `loss.decision.*` added later
  with `weight: 0.0` (inactive), `trainer.max_time` null (set at launch). Nothing else differs.
- Per-scale launch overrides (the §2 step rule, nothing else): `trainer.max_steps=81068`, `trainer.max_epochs` large
  enough that the step cap binds (the cosine total is then `max_steps`, `train_lewm_mv.py:348-350`; W1's total was
  100 × len(train) ≈ 81,1xx, a ≤ 0.05 % schedule difference, logged as such), `trainer.max_time` above the run's
  expected time so the step cap, not the clock, ends training. Split file = the scale's list below.

## C3 — S1 / S2 lists (2026-11-02)

`splits/S1/split.json` (45), `splits/S2/split.json` (90) (named `split.json`: `vast/launch.sh` copies the file by name and `run_remote.sh` reads `split.json`): first 45 / 90 of `numpy.random.default_rng(0).permutation(train_episodes)`
of `wm_train/split.json` (numpy 1.23.5, the permutation stored in each file); val episodes unchanged (25); S1 ⊂ S2
asserted. FACTS.json entries for the three datasets, the S5b count: still open (needed before S4 / S5).

## C3 — the three datasets, S4 / S5a lists (2026-11-02, `c3_meta.py` → `c3_meta.json`, FACTS keys of the same names)

| dataset | revision | episodes | frames | fps |
|---|---|---|---|---|
| `so101_bowls_all_trimmed` | `4e5c6fda8dc5` (W1's split) | 182 | 61,462 | 30 |
| `so101_cubes_bowls_20261015` (private) | `671ee2263683` | 653 | 222,000 | 30 |
| `armnet/armnetbench_v01_lerobot_so101` | `1db69c82b589` | 2,499 (2,099 rollouts + 400 teleop) | 1,127,881 | 20 |

- **S5b count = 515** ArmnetBench rollouts with `success` true that are not teleop (`c3_meta.json` `s5b_success_rollouts`),
  equal to the memo's 915 − 400.
- Training frames per scale (native rate, the §2 x-axis): S1 15,032 · S2 30,469 · S3 53,015 · S4 275,015 · S5a 1,282,161
  · S5b 404,730 (`c3_meta.json` `train_frames_native`; the ArmnetBench part is at 20 Hz until the C4 rate rule is set).
- Lists: `splits/S4/split.json` (157 bowls + 653 cubes = 810) and `splits/S5a/split.json` (+ 2,099 ArmnetBench rollouts
  = 2,909), built by `make_union_splits.py`: W1's split plus `extra_train` sources (`wm_train/data_union.py`); val = the
  same 25 bowls episodes.

## C4 — ArmnetBench mapping (2026-11-02, `c4_mapping.py` → `c4_mapping.json`, `c4_frames.png`, `c4_joints.png`)

Measured on ArmnetBench episodes 3 (diffusion, successful) and 0 (diffusion, failure), the lowest-index rollout of each
outcome, against our bowls episode 0.

| item | finding | in the S5a split |
|---|---|---|
| cameras | ours `front_top_camera` is an overhead-oblique view of the table; ArmnetBench `top` is the matching view (`front` is low and side-on); `wrist` ↔ `wrist_camera` (gripper fingers at the bottom in both) | `top` → pixels, `wrist` → wrist; `front` not loaded |
| resolution / codec | ArmnetBench top 1024×576, wrist 1280×720, AV1; ours 640×480 AV1 | resized to 224 by the loader, as ours (aspect 16:9 vs 4:3 squashed differently) |
| joint units | **all three datasets are in degrees**: cubes `wrist_roll` reaches 165.4, beyond the ±100 range mode but inside the calibrated ±167.0°; bowls and cubes rest poses (first frames, same rig) agree as recorded within 3.2 on every joint, and disagree by up to 5.1 if bowls were range mode; ArmnetBench is `use_degrees` (`data/community/REPORT.md:118`) | no conversion; gripper 0–100 in all |
| frame rate | ours 0.0333 s, ArmnetBench 0.05 s per frame (timestamps) | **open: the rate rule (owner)** |

Rate rule, two options (the protocol names the rule, not its form; nothing chosen):
(a) **native 20 Hz**: windows of 8 ArmnetBench frames span 0.40 s vs 0.27 s for ours, with no rate signal to the model; no
preprocessing; the loader reads the Hub copy as is (`smoke_union.py` passed on 2 episodes per source, `frameskip 1`).
(b) **resample to 30 Hz**: frame k at 30 Hz = native frame ⌊2k/3⌋ (every third frame repeated, actions likewise); needs a
preprocessed copy (memo gap 6: ≤ $1.5 on a vast CPU box, not run).
Raw read cost for S5a on the training box either way: top + wrist videos 43.2 GB of the 60.5 GB (Hub tree sizes), decoded at
1.9–3× our pixels per frame; the step time on it is unmeasured (a 224 transcode, as in the memo, removes that risk).

## C5 — held-out episodes (S1, S2)

Asserted when the lists were built: no val episode in S1 / S2; every listed episode is in the 157 train episodes. The 69
rollouts are another dataset (`ravediamond/so101-bowls-policy-rollouts`), not loaded by a split of
`so101_bowls_all_trimmed`.

C5 for S4 / S5a: the bowls part is W1's split (no val episode in it, asserted by `make_union_splits.py`); cubes and
ArmnetBench are other repos.

Recipe diff added by the loader (2026-11-02): config key `extra_root` (null unless the split has `extra_train`); S1 / S2 /
S3 runs read no extra source.

Open before their numbers: C2 (probes.py reproduction), C4 rate rule (before S5a), C6, C7.

## 2026-11-06 — C5 on the shipped splits, C7, C1 end steps

- **C1 end steps:** S1, S4, S5a `metrics.jsonl` end record `"step": 81068` (`~/robot-data/wm/runs/<run>/out/multiview/metrics.jsonl`;
  runs 20261103T1854_WS-106_59001611, 20261104T0657_WS-106_59070325, 20261104T1744_WS-106_59139781).
- **C5:** the `split.json` each instance trained on (`out/multiview/split.json`): 0 of the 25 val episodes in any train list, the 25
  val episodes listed as val, extras = cubes (S4) and cubes + ArmnetBench (S5a), no rollouts repo (`c5_check.txt`).
- **C7:** the §30 random-init encoder is `id_latents.py` `W1Like(<w1>, True)` (same config, seed 0; README of
  `shared/gate/data/2026-10-24_inverse_dynamics`), its latents `~/robot-data/id30_latents/<set>/W1_random_init.npz`;
  `probes.py p3lat` links them unchanged into every scale's latent dir.
- **C2** runs through `probes.py c2 runs/WS-106/probes/S3` (`run_all.sh` stops the other scales on a mismatch).
