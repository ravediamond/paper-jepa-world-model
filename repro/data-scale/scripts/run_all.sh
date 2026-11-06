#!/usr/bin/env bash
# WS-106: C2 on W1 (S3) first; the other scales run only if every C2 value matches (PROTOCOL.md §6: a mismatch stops).
#   nohup bash projects/jepa-data-scale/run_all.sh > runs/WS-106/run_all.out 2>&1 &
set -uo pipefail
REPO=$HOME/workspace/road-to-robotics; cd "$REPO"
R=$HOME/robot-data/wm/runs

bash projects/jepa-data-scale/run_battery.sh S3 /dev/null p4 p3 p5 && python3 projects/jepa-data-scale/probes.py c2 runs/WS-106/probes/S3 || { echo "C2 MISMATCH: stopped"; exit 1; }
for s in "S1 $R/20261103T1854_WS-106_59001611" "S4 $R/20261104T0657_WS-106_59070325" "S5a $R/20261104T1744_WS-106_59139781"; do
  set -- $s
  bash projects/jepa-data-scale/run_battery.sh "$1" "$2/out/multiview/weights_final.pt"
done
echo "all scales done"
