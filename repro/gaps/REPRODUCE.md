# Reproduce

## L0
```bash
python3 l0_check.py
```
Checks the orderings and means the paper states from `expected/controls.json`.

## L1
`scripts/extract_controls.py` (W1 latents need the strict-load image, transformers 5.19.0) then `scripts/analyze_controls.py`, on the rollouts dataset pinned in `data/DATA.md`.

## L2
Fresh rollouts on the arm, scored under a bar committed first (not done).
