# Reproduce

## L0
```bash
python3 l0_check.py
```
Recomputes the four verdicts from the stored probe outputs and checks the P2 floor, the P3 three-part rule, the P4 and P5 statements, the training frames and the split lists. Standard library and numpy.

## L1
Needs the four checkpoints (`weights.sha256` in each `expected/<scale>/`) and the working repository `ravediamond/road-to-robotics` (private): `scripts/run_battery.sh <scale> <weights_final.pt>` runs P1 to P5 on a checkpoint, and `scripts/probes.py verdict <dir>` joins them. About 2.5 hours per scale on an Orin NX; the P3 ridge dominates it.

## L2
Retraining a scale took 22.3 h (S1), 9.9 h (S4) and 10.8 h (S5a) on rented GPUs (`elapsed_s` in each `training_end.json`; the machines differed). Not repeated for this folder.
