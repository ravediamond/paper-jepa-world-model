#!/usr/bin/env bash
# WS-106 probe battery on one checkpoint (PROTOCOL.md §3), Orin GPU, $0. Results in runs/WS-106/probes/<scale>/.
#   bash projects/jepa-data-scale/run_battery.sh <scale> <weights_final.pt> [p1 p2 p3 p4 p5]
# The container is the W1 one (shared/gate/jetson_run.sh W1=1: lerobot-jetson-wm:rc + pydeps-w1, strict load) plus
# read-only mounts of ~/robot-data (data, checkpoints) and the paper-vs-hardware post (gate0_extract's split / probe).
set -euo pipefail
S=$1; W=$2; shift 2
STEPS=${*:-p1 p2 p4 p3 p5}   # p4 first: it downloads the rollouts snapshot that p3 reads
REPO=$HOME/workspace/road-to-robotics
OUT=runs/WS-106/probes/$S
STAGE=$HOME/wm_data/ws106/$S           # E.load reads <dir>/weights_final.pt; ~/wm_data is the W1 mount
DATA=$HOME/robot-data/so101-bowls-h264-224
HUB=$HOME/.cache/huggingface/hub/datasets--ravediamond--so101-bowls-policy-rollouts
# rollouts: the Hub snapshot run_w1_offline_screening downloads (main; same data blobs as FACTS revision bad1c8e, only README.md differs)
roll() { echo "/root/.cache/huggingface/hub/datasets--ravediamond--so101-bowls-policy-rollouts/snapshots/$(cat "$HUB/refs/main")"; }
LABELS=data/rollouts/so101-bowls-policy-rollouts_labels.csv
mkdir -p "$STAGE" "$REPO/$OUT"
[ -f "$STAGE/weights_final.pt" ] || cp "$W" "$STAGE/weights_final.pt"
sha256sum "$W" "$STAGE/weights_final.pt" > "$REPO/$OUT/weights.sha256"
say() { echo "$(date '+%F %T') [$S] $*" | tee -a "$REPO/$OUT/battery.log"; }
dock() {
  docker run --rm -i --runtime nvidia --ipc host --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$HOME/wm_data:$HOME/wm_data:ro" -v "$HOME/gate/pydeps-w1:/opt/pydeps-w1:ro" \
    -v "$HOME/gate/pydeps:/opt/gate-pydeps:ro" -v "$HOME/robot-data:$HOME/robot-data:ro" \
    -v "$HOME/workspace/git/paper-vs-hardware/posts:/pvh/posts:ro" \
    -v "$REPO:/work" -w /work -e PYTHONUNBUFFERED=1 \
    -e WM_SPLIT=/work/projects/vla-jepa-world-model/wm_train/split.json \
    -e PYTHONPATH="/opt/pydeps-w1:/work/shared/gate:/opt/gate-pydeps" \
    lerobot-jetson-wm:rc \
    bash -c '"$@"; rc=$?; chown -R '"$(id -u):$(id -g)"' /work/runs/WS-106 2>/dev/null; exit $rc' _ "$@"
}
PS=projects/jepa-data-scale/probes.py
for st in $STEPS; do
  say "start $st"
  case $st in
    p1) dock python projects/lejepa-identifiability/scripts/gate0_extract.py --data "$DATA" --w1 "$STAGE/weights_final.pt" \
          --out "$OUT/p1" > "$REPO/$OUT/p1.log" 2>&1
        python3 projects/lejepa-identifiability/scripts/gate0_analyze.py "$REPO/$OUT/p1" "$REPO/$OUT/p1" >> "$REPO/$OUT/p1.log" 2>&1 ;;
    p2) dock python $PS p2 "$STAGE" --data "$DATA" --out "$OUT/p2.json" > "$REPO/$OUT/p2.log" 2>&1 ;;
    p3) dock python $PS p3lat "$STAGE" --teleop "$DATA" --rollouts "$(roll)" --outcomes "$LABELS" \
          --ref "$HOME/robot-data/id30_latents" --out "$OUT/p3lat" > "$REPO/$OUT/p3.log" 2>&1
        if [ "$S" = S3 ]; then P3=(projects/vla-jepa-world-model/wm_train/id_probe.py); else P3=($PS p3probe); fi
        mkdir -p "$REPO/$OUT/p3"
        if [ "$S" = S3 ]; then python3 "${P3[@]}" --latents "$REPO/$OUT/p3lat" --out "$REPO/$OUT/p3/results_ridge.json" >> "$REPO/$OUT/p3.log" 2>&1
        else python3 "${P3[@]}" "$REPO/$OUT/p3lat" --out "$REPO/$OUT/p3/results_ridge.json" >> "$REPO/$OUT/p3.log" 2>&1; fi ;;  # p3probe takes the latents dir positionally (2026-11-06 fix: the first run of S1, S4, S5a stopped here)
    p4) dock python scripts/run_w1_offline_screening.py --w1 "$STAGE/weights_final.pt" --out "$OUT/p4" > "$REPO/$OUT/p4.log" 2>&1 ;;
    p5) python3 $PS p5 "$REPO/$OUT/p4/latents" --out "$REPO/$OUT/p5.json" > "$REPO/$OUT/p5.log" 2>&1 ;;
  esac
  say "done $st rc=$?"
done
python3 $PS verdict "$REPO/$OUT" > /dev/null && say "verdict written"
