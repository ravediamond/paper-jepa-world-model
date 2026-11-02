# make check: every L0 check, then the paper regenerated from repro/*/expected and compared with the committed
# numbers.json / main.tex / DRAFT_4p.md (any difference fails), then the PDF rebuilt.
PY ?= python3
export SOURCE_DATE_EPOCH := 1761955200
TECTONIC ?= tectonic
check: l0 paper pdf
l0:
	@for d in repro/*/; do echo "== $$d"; (cd $$d && $(PY) l0_check.py | tail -1 | grep -q "L0 PASS") || { echo "L0 failed in $$d"; exit 1; }; done
paper:
	$(PY) paper/make_paper.py
	git diff --exit-code -- paper/numbers.json paper/main.tex paper/DRAFT_4p.md
pdf:
	git rev-parse --short HEAD | sed 's/.*/\\shissocommit{&}/' > paper/shisso_commit.tex
	cd paper && $(TECTONIC) main.tex
.PHONY: check l0 paper pdf
