# Files copied from paper-register-reproducibility

Verbatim copies, read by `paper/make_paper.py`, so that the numbers this paper shares with
`ravediamond/paper-register-reproducibility` are the same numbers (rule 100). Source commit: `f554775`.

| file here | source path there | what this paper reads |
|---|---|---|
| `bootstrap.json` | `repro/baseline-bootstrap/expected/bootstrap.json` | paired bootstrap of copy-last (C-001-1), chunk ranking (C-001-2), RoboJEPA (C-003-1), ContactGuard (C-006-1), ReAP (C-008-1) |
| `conformal_check.json` | `repro/conformal-check/expected/conformal_check.json` | the conformal band built as the papers build it, and the replay of our first (wrong) band |
| `reap_results.json` | `repro/variants/reap/expected/results.json` | ReAP Stage 1 + Stage 2 rebuilt from the paper, main set, seed 0 |
| `datasets.json` | `data/datasets.json` | rollout main set (50 / 42 / 8), teleop frames and fps |
| `register.json` | `repro/register/expected/register.json` | ContactGuard's own numbers (claim C-006-1, `theirs`) |

The protocols, scripts and L0 checks of these results stay in the source repository.
