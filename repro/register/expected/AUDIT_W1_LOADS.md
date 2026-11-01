# Checkpoint-load audit of every W1 / W2 / W2′ gate (WS-091 step 0, 2026-11-01)

Why: G11-A (2026-10-18) timed a **random** ViT encoder — `strict=False` with a key remap that did not fit
transformers 5.5.4 (the voided-gates post). This audit asks the same question of every gate that used a world-model
checkpoint: was the encoder that produced the number the trained one?

Method: for each gate, the script that produced its number (from the gate row's source folder), the call that builds
the model, whether that call is strict, and the environment. Rule: a **strict** load (PyTorch's `load_state_dict`
default, or `so101_harness.w1_live.load_state_dict_strict`) either loads every trained tensor or raises — in an
environment with the wrong transformers version it raises (shown 2026-11-01: 192 missing / 192 unexpected keys under
5.5.4, all 279 tensors under 5.19.0, `paper-vs-hardware/posts/2026-10-22-voided-gates`). So a strict load that ran to a
result is **trained**. A non-strict load is **random** if keys were dropped, **unknown** if nothing shows they were not.
W2 / W2′ encode with a frozen pretrained DINOv2-S from the Hub (`AutoModel.from_pretrained`), trained by construction;
their predictors load with the strict default.

Load paths found (road-to-robotics, `projects/vla-jepa-world-model/` unless noted):

| path | builds the model with | strict | used by |
|---|---|---|---|
| A | `wm_train/eval_action_sensitivity.load` (`E.load`) → `model.load_state_dict(sd)` | yes (default) | `probe.py`, `g1b_range.W1`, `g1_suite`, `action_sensitivity.W1Like`, `g2_orin_budget`, `hwm_extract`, LeJEPA `gate0_extract`, `djepa_operator` (via `g1b_range.W1`), `g7a_score` (same), `id_latents` |
| B | `wm_train/g10_eval.load_w1_model`, `wm_train/g6b_eval.load_w1_model` → `load_state_dict(sd)` | yes (default) | G10, G6b |
| C | `src/so101_harness/w1_live.W1LiveMonitor.from_checkpoint` → `load_state_dict_strict` (strict, one strict retry with the remap; never `strict=False`; added in `787aadf`, 2026-10-18, with the class itself) | yes | every W1LiveMonitor script: W1-A/B, G11-B-cal, W1-C1/C2, REAP (steps 4, v1, v2 latents), SURVEYOR, §29 scoring, the live runners |
| D | `jetson/bench_w1_surprise_orin.py` → `load_state_dict(remapped, strict=False)` | **no** | G11-A (timing) |
| E | `jetson/run_g11_phase_b_live.py` → `strict=False` | **no** | G11 phase B — **never run** |
| F | LeRobot `PreTrainedPolicy.from_pretrained` (`strict=False` default, `lerobot/policies/pretrained.py`) | **no** | VLA-JEPA (its own world predictor, not W1) |

## Table

| gate (date) | script | checkpoint | path | env | evidence the encoder was trained | verdict |
|---|---|---|---|---|---|---|
| G0 (10-11) | `wm_train/eval_action_sensitivity.py` | W1 run dir (= Hub `lewm-w2-so101-bowls` `w1/` @ `d0336db6`, uploaded 10-26) | A | training env (vast / Mac, the version that saved W1) | strict load; corruption ratios ≥ 1.3 and open loop 0.31 vs copy-last 0.89 | trained |
| C1 probe (10-11) | `wm_train/probe.py` | W1 | A | Mac | strict load | trained |
| G1b W1 (10-12) | `wm_train/g1b_range.py` | W1 | A | Mac | strict load | trained |
| G1 probes (10-12) | `wm_train/g1_suite.py` | W1, W2 | A / DINOv2 | Mac | strict load; W2 frozen pretrained | trained |
| G2 Mac (10-12) | `wm_train/g2_orin_budget.py` | W1, W2 | A | Mac | strict load (timing gate) | trained |
| G1 W2 dynamics (10-12) | `wm_train/g1_suite.py` | W2 | DINOv2 + strict predictor | Mac | pretrained encoder | trained |
| G6a (10-14) | `wm_train/g6a.py` → probe latents | W1, W2 | A | Mac | strict load | trained |
| G10 (10-17) | `wm_train/g10_eval.py` | W1, W2 | B | Mac | strict load; readout AUROC 1.000 | trained |
| G6b (10-18) | `wm_train/g6b_eval.py` | W1 (+ H-JEPA level 2 trained on its latents) | B | Mac | strict load | trained |
| **G11-A (10-18)** | `jetson/bench_w1_surprise_orin.py` | W1 | **D** | Orin, image 5.5.4 | strict-load check of the same day: 192 encoder tensors dropped | **random** — already voided (EXPERIMENTS 10-18, "timing valid, latents void"); timing stands |
| W1 still-scene noise floor (10-18) | `2026-10-18_w1_noise_floor/strict_load_log.py` | W1 | C | Orin, 5.19.0 overlay | the strict-load log itself (load only, no value) | trained (no number) |
| W1-A, W1-B (10-19) | `scripts/run_w1_offline_screening.py` | W1 | C | Orin, `W1=1` (5.19.0) | strict load | trained |
| G11-B-cal (10-19) | `scripts/run_g11b_cal.py` | W1 | C | Orin, `W1=1` | strict load | trained |
| W1-C1, W1-C2 (10-19) | `scripts/run_w1c_offline.py` | W1 | C | Orin, `W1=1` | strict load | trained |
| W1-R (10-22) | `wm_train/djepa_operator.py` (latent cache from `g1b_range.W1`) | W1 | A | Mac / Orin | strict load; the step-0 check reproduces W1's G1b top-1 (`w1r.json` `step0.reproduced: true`) | trained |
| REAP step 4 (10-22), REAP monitors v1 (10-23) | `scripts/run_reap.py` | W1 | C | Orin, `W1=1` | strict load | trained |
| REAP v2 (10-23) | `scripts/run_reap_v2.py` on REAP's latents | W1 | C | Orin | strict load (latents from `run_reap.py`) | trained |
| W2′ re-train (10-23) | `wm_train/train_w2.py` + `g1_suite.py` | W2′ | DINOv2 + strict | vast / Mac | reproduces W2's G1 numbers | trained |
| §28 action sensitivity (10-25) | `wm_train/action_sensitivity.py` | W1, W2′ | A | Orin | strict load | trained |
| §30 inverse dynamics (10-26) | `wm_train/id_latents.py` (`W1Like`) | W1 @ `d0336db6`, random-init W1 (explicit control), W2′ | A | Orin | strict load for W1; the random-init row is the protocol's own control, built separately | trained (the "random-init matches W1" result is genuine) |
| LeJEPA gate 0 (10-27) and correction (10-28) | LeJEPA `gate0_extract.py` | W1 @ `d0336db6`, DINOv2-S | A | Orin, W1 env | strict load; the correction fixed an input-indexing bug (red channel), not the load | trained |
| SURVEYOR gates 0–1 (10-27) | `projects/surveyor/scripts/surveyor_imagine.py` | W1 | C | Orin | strict load | trained |
| HWM gate 0 (10-28) | `projects/hwm-crossover/scripts/hwm_extract.py` | W1 | A | Orin, W1 env | strict load | trained |
| G7a pipeline test | `wm_train/g7a_score.py` (`g1b_range.W1` / W2) | W1, W2 | A | Mac | strict load; marked `PIPELINE_TEST_NOT_G7A` | trained (not a gate result) |
| G11 phase B (live) | `jetson/run_g11_phase_b_live.py` | W1 | **E** | — | never run | n/a |
| VLA-JEPA surprise, §29 steps 0–1 (10-25) and robot day (10-27) | VLA-JEPA's own world predictor via LeRobot | `ravediamond/vla_jepa_so101_bowls_trimmed` (§29: @ `c65c98b`) | **F** | Orin | non-strict load; indirect evidence: on the same checkpoint the legacy-code world-model loss beats copy-last by ≈ 1.06 L1 (rope-compat post, 200 windows), which a missing or random predictor could not; the action head runs 9/10 and 20/20 on the arm | trained (indirect; no key-set log) |

## Counts

**Trained: 26 rows** (one, the noise floor, produces no number; G7a is a pipeline test) · **random: 1** (G11-A
latents, already voided; its timing stands) · **unknown: 0** · not run: 1 (G11 phase B). One row rests on indirect
evidence (VLA-JEPA, a non-strict LeRobot load). **No paper claim rests on a row that is not `trained`.**

## Reruns and checks

- Random rows to rerun: none beyond G11-A, which carries no claim (its latents were never used after 10-18).
- Cheap hardening ($0, offline): log the key sets of the VLA-JEPA checkpoint load once (`missing_keys`,
  `unexpected_keys` of `load_state_dict(..., strict=False)`'s return) to turn the indirect row into a direct one.
- Known-output check (WS-090, 2026-11-01): LeJEPA gate 0 rerun cold in the published W1 image
  (`ghcr.io/ravediamond/lerobot-jetson:w1`, W1 from the public Hub @ `d0336db6`): W1 mean R² 0.5654, anisotropy 0.6393,
  G1b 0.6046 and W2′ 0.8672 / 0.3007 / 0.6458 — **identical** to the published `gate0.json`; voided-gates' strict-load
  log also identical under both transformers versions (`paper-vs-hardware/rig/w1/L1_W1.md`).
