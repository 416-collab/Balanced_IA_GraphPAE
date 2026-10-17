#!/bin/bash
set -u

cd ~/GraphPAE_Project/repo/GraphPAE-8a8cd33

OUTDIR=~/GraphPAE_Project/results_server/IA_RERUN_5SEED
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS="chameleon squirrel actor cora citeseer"
RATIOS="0.25 0.10"
NUMEXP=5

run_one () {
  DATASET=$1
  RATIO=$2

  echo ""
  echo "============================================================"
  echo "IA RERUN: dataset=$DATASET ratio=$RATIO num_exp=$NUMEXP"
  echo "============================================================"

  python set_budget_config.py \
    --dataset "$DATASET" \
    --sampler ia \
    --ratio "$RATIO" \
    --tag ia_rerun

  LOG="$OUTDIR/${DATASET}_ia_ratio${RATIO}_numexp${NUMEXP}_rerun.txt"

  python -u train_node.py --dataset "$DATASET" --num_exp "$NUMEXP" 2>&1 | tee "$LOG"

  RC=${PIPESTATUS[0]}
  echo "Return code: $RC"
  echo "Saved log: $LOG"

  if [ $RC -ne 0 ]; then
    echo "FAILED: dataset=$DATASET ratio=$RATIO"
    exit $RC
  fi
}

for D in $DATASETS; do
  for R in $RATIOS; do
    run_one "$D" "$R"
  done
done

python parse_ia_rerun.py

echo "IA rerun completed."
