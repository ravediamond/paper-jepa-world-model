# Owner decisions on PROTOCOL.md §9 (2026-11-01, before any training; chat on behalf of the owner)

- **O1** P5 floor = the G9c-3 judge score (the no-learning reference, 0.943 on gap 2); W2′ and joints reported beside it.
- **O2** P2 wording: "for each scale, top-1 at the largest d where it is measured; the probe clears the floor at that
  scale if top-1 − action-only floor > 0 with CI > 0 at that same d" (floor replaces 0.8; 0.8 reported as the original bar).
- **O3** P4 floor = W2′'s **same** signal (L2 surprise), paired by episode, each at its own in-sample τ; with 8 failures
  report the count difference and its bootstrap CI and say the CI is coarse; no pass/fail on P4, descriptive.
- **O4** Same cell for bar and floor (P2: same d; P3: same k). Single-seed scales count for the curve but not for the
  falsification rule; the rule reads S1 and S5a (two seeds each). No multiplicity correction (pre-registered, reported
  as a family of five probes; say so). "Smallest scale" = the smallest scale from which every larger scale also clears
  (G1_PROTOCOL.md:43 form); non-monotone cases reported as measured in a note.
- **O5** S5b (successes only) and S4-fixed-epochs: exploratory beside the headline; the headline reads "by S5a".
- **O6** W2′ at seed 3073 at every scale; random-encoder predictor seed 0; controls reran at seed 0 at S1 / S5a only.

## Amendment 2 — 2026-11-02 (owner), before any WS-106 number exists

**A2-1. Random-init encoder control.** The probe battery gets one more control at **every scale**: the **random-init encoder** (the same architecture, untrained, fixed seed), run through the same probes on the same rows as the trained model at that scale, so each scale's row reports trained vs random-init. Reason (WS-110 gap 5, `projects/jepa-distill-edge/results/gap5.json`, exploratory): on the §30 term the trained W1 scored 0.7474 against 1.4383 for its own random init; a trained model below its untrained version is only visible if the control is there at every scale. No bar, threshold or decision rule of the frozen protocol changes; the control is reported beside the existing rows and is not part of any existing bar.

**Correction note (dispatcher, 2026-11-02; no bar changes).** A2-1's reason cited "trained W1 0.7474 against 1.4383 for its own random init". The 1.4383 was BLAS-dependent (ill-conditioned ridge):
the same untrained network scored 0.7974 and 0.8042 on two other machines (`projects/jepa-distill-edge/results/gap5_pooled_rerun.json`), so the trained W1 sits at its random-init floor, not below it
(W1 − random-init −0.0499, CI [−0.7403, +0.6559]). The control (the random-init encoder at every scale) stands and is, if anything, more necessary: a probe term can move this much with the BLAS build.
Add to the control's reporting: run the random-init control's ridge with the same solver and report the condition number of its design matrix.

## Amendment 3 — 2026-11-02 (owner), before any S4 / S5a number exists

**A3-1. C4 rate rule = (a): ArmnetBench stays at its native 20 Hz, no resampling** (CHECKS.md C4: ours 0.0333 s, ArmnetBench 0.05 s per frame). **Reporting labels (the owner's words):** S5a is reported as **"scale + ArmnetBench at native 20 Hz"**, and **S4 as the clean same-rate point** (our own data only, 30 Hz): the curve's S4 → S5a step therefore changes the data scale **and** the frame rate of part of the data, and the report says so wherever S5a appears.
**A3-2. The ArmnetBench 224-px transcode is funded: resolution only (no frame-rate change, no resampling in time), vast CPU, ≤ $1.50** (cost line on the item before launch; bandwidth included; WS-127 path; rows / the transcoded files returned, verified by the manifest). **Order after the S4 retry: transcode → S5a** (S5a launches only by the credit rule: credit − worst case ≥ $6.50 and the running total of WS-106 worst cases ≤ credit − $6.50).
No bar, threshold or decision rule of the frozen protocol changes.
