#!/bin/bash
set -u

cd ~/GraphPAE_Project/repo/GraphPAE-8a8cd33

OUTDIR=~/GraphPAE_Project/results_server/IA_BUDGET_SCREEN
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${1:-chameleon}
RATIO=${2:-0.25}
NUMEXP=${3:-1}

SAMPLERS="random spectral degree ia"

for S in $SAMPLERS; do
  echo ""
  echo "============================================================"
  echo "SCREEN: dataset=$DATASET sampler=$S ratio=$RATIO num_exp=$NUMEXP"
  echo "============================================================"

  python set_budget_config.py --dataset "$DATASET" --sampler "$S" --ratio "$RATIO" --tag screen

  LOG="$OUTDIR/${DATASET}_${S}_ratio${RATIO}_numexp${NUMEXP}_screen.txt"

  python -u train_node.py --dataset "$DATASET" --num_exp "$NUMEXP" 2>&1 | tee "$LOG"

  RC=${PIPESTATUS[0]}
  echo "Return code: $RC"
  echo "Saved: $LOG"

  if [ $RC -ne 0 ]; then
    echo "FAILED during selector screening."
    exit $RC
  fi
done

python parse_selector_screen.py 2>/dev/null || true
