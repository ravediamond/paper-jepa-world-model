# M3 sanity failure: diagnosis (excerpt)

Verbatim from road-to-robotics `projects/jepa-surprise-receptive-field/README.md` lines 38-39 and 49-53 @ 7f93035; the per-anchor numbers are in `M3_diag_action_shuffle.json`.

M3's class-5 AUROC is 0.501 against the 0.9 sanity bar (`results/M3_table.md` l.36). The $0 diagnosis (owner `WS-116 b`; `results/M3_diag_action_shuffle.json`, `scripts/` as logged): the world-model forward **does take the action tokens as an input**
(token tensor [1, 24, 2048]); on 5 anchors the loss was computed with the true tokens, with tokens taken from another episode (a fixed derangement, seed 116), and with the tokens zeroed. Repeated forwards are bit-identical (noise 0.0).
*Exploratory, 5 anchors: a spot check, not a power analysis.* **VLA-JEPA's predictor is near action-blind:** shuffled actions change the action tokens by 21–28 % (relative L2) and the loss by 0.05 % on average
(largest 0.09 %, mixed sign); zeroing the tokens changes it by 1.1 % on average (0.4–2.2 %); the forward is deterministic. M3's own class-5 shift in the scored run
(0.000675) is the same size as the shuffle shift (0.000697, mean |Δ| over the mean loss 1.4352).
Per the owner's ruling (r3) the shift is treated as noise-sized: **0.501 is a real value, M3's table stays unread** per the frozen bar (amendment O1: it blocks only M3's table); no rescoring.
