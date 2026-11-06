# FREEZE — WS-106 protocol, 2026-11-02 (before any number)

Frozen set: `PROTOCOL.md` (de586b8) + `AMENDMENT_2026-11-01.md` (O1–O6 at 106ae73; Amendment 2 at c69e57e; correction
note at abd8590). This file only maps the owner's answers onto the protocol numbering. **No bar, threshold or decision
rule is written or changed here**; where an answer and a protocol line differ, the amendment (owner, later, before any
number) is the one that reads.

| O | protocol line it settles | reads as (amendment) |
|---|---|---|
| O1 | §4 P5 row "floor: **open (O1)**"; §9 O1 | P5 floor = G9c-3 judge score (0.943 on gap 2, as quoted in the amendment); W2′ and joints reported beside it; G10 0.90 stays alongside (§5 P5) |
| O2 | §5 P2 "Floor rule … *Wording to confirm (O2)*"; §9 O2 | per scale, top-1 at the largest d where it is measured; clears the floor at that scale if top-1 − action-only floor > 0 with CI > 0 at that same d; 0.8 reported as the original bar |
| O3 | §4 P4 row (floor = W2′ at the same scale); §5 P4; §9 O3 | floor = W2′'s **same** signal (L2 surprise), paired by episode, each at its own in-sample τ; count difference + bootstrap CI, stated as coarse; **P4 descriptive, no pass / fail** — so in the §5 decision rule ("one of P2–P4 clears both its original bar and its floor") P4 is reported, not counted |
| O4 | §5 decision rule (same cell, seeds, "smallest scale"); §9 O4 | bar and floor at the same cell (P2 same d, P3 same k); single-seed scales (S2, S3, S4, S5b) on the curve, not in the falsification rule, which reads S1 and S5a (two seeds each); no multiplicity correction, reported as a pre-registered family of five probes; "smallest scale" = smallest from which every larger scale also clears (`wm_train/G1_PROTOCOL.md:43` form), non-monotone cases in a note |
| O5 | §5 headline "by S5a"; §8 exploratory list; §9 O5 | S5b and S4-fixed-epochs exploratory beside the headline; headline reads "by S5a" (matches §8 as written) |
| O6 | §4 P1/P2/P4 control rows; §2 seeds; §9 O6 | W2′ at seed 3073 at every scale; random-encoder predictor seed 0; control reruns at seed 0 at S1 / S5a only |
| A2-1 | §4 "also reported" column (random encoder at S1 / S3 / S5a in the P4 row) | random-init encoder (same architecture, untrained, fixed seed) at **every** scale, same probes, same rows; reported beside, part of no bar |
| A2 note | reporting of the A2-1 control | same ridge solver as the trained model; report the design-matrix condition number |

Not settled by the amendment (no bar involved; reported as written, flagged here): §10's cost line has the random
encoder at 3 × $0.65; A2-1 puts the control at every scale. The encoder itself is untrained (probes on the Orin, $0);
whether P4's random-encoder predictor needs training at the 3 extra scales (≈ 3 × $0.65 more) is a cost question for
the launch plan, not a protocol field.
