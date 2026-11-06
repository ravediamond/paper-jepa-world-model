#!/usr/bin/env bash
# WS-106, 2026-11-06: the first battery of S1, S4 and S5a stopped at P3 (p3probe takes the latents dir positionally; run_battery.sh fixed).
# This runs P3 probe, P5 and the verdict for one scale from the stored p3lat / p4 latents, CPU only, detached.
#   for s in S1 S4 S5a; do nohup bash projects/jepa-data-scale/finish_scales.sh $s > runs/WS-106/finish_$s.out 2>&1 & done
set -uo pipefail
S=$1; REPO=$HOME/workspace/road-to-robotics; cd "$REPO"; O=runs/WS-106/probes/$S; PS=projects/jepa-data-scale/probes.py
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
say() { echo "$(date -Is) [$S] $*" | tee -a $O/battery.log; }
mkdir -p $O/p3
say "start p3probe"; python3 $PS p3probe $O/p3lat --out $O/p3/results_ridge.json > $O/p3.log2 2>&1; say "done p3probe rc=$?"
say "start p5"; python3 $PS p5 $O/p4/latents --out $O/p5.json > $O/p5.log 2>&1; say "done p5 rc=$?"
python3 $PS verdict $O > /dev/null 2>&1; say "verdict rc=$?"
