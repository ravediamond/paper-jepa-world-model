# A small JEPA world model knows what happened, not what will

Every pre-registered use of one LeWM-style JEPA world model on one SO-101 arm: the encoder carries the outcome, every forecasting use of the predictor fails. Draft: `paper/DRAFT_4p.md` / `paper/main.pdf`.

## What is here

- `paper/`: the draft (markdown and LaTeX sources, `make_paper.py`, `numbers.json` with every number's anchor, generated `fig/`, the built `main.pdf`).
- `repro/`: one folder per experiment the paper cites (gaps, register): protocol, scripts, the expected result files the paper reads, `l0_check.py`, `REPRODUCE.md`.
- `data/DATA.md`: every dataset and checkpoint, Hub id and pinned revision. No raw data is stored here.
- `PROVENANCE.md`: where every file was copied from.

## Reproduce

- **L0** (CPU, seconds): `make l0` — recompute every number from `repro/*/expected/`.
- **Paper**: `make check` — L0, then `paper/make_paper.py` regenerates `numbers.json` and the sources from the expected files and fails on any difference, then rebuilds the PDF (needs `tectonic`, numpy, matplotlib).
- **L1** (rerun on the stated hardware) and **L2** (the arm): per experiment, in `repro/<exp>/REPRODUCE.md`.

## Cost

W1 training $3.30 on a rented GPU (register row G0); everything here is $0 on the Orin CPU.

## Licence

Code MIT; text and figures CC BY 4.0 (to confirm before any release).
