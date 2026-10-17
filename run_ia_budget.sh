#!/bin/bash
set -u

cd ~/GraphPAE_Project/repo/GraphPAE-8a8cd33

OUTDIR=~/GraphPAE_Project/results_server/IA_BUDGET_RUN
mkdir -p "$OUTDIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODE=${1:-smoke}

run_one () {
  DATASET=$1
  SAMPLER=$2
  RATIO=$3
  NUMEXP=$4
  TAG=$5

  echo ""
  echo "============================================================"
  echo "Running dataset=$DATASET sampler=$SAMPLER ratio=$RATIO num_exp=$NUMEXP tag=$TAG"
  echo "============================================================"

  python set_budget_config.py --dataset "$DATASET" --sampler "$SAMPLER" --ratio "$RATIO" --tag "$TAG"

  LOG="$OUTDIR/${DATASET}_${SAMPLER}_ratio${RATIO}_numexp${NUMEXP}_${TAG}.txt"

  python -u train_node.py --dataset "$DATASET" --num_exp "$NUMEXP" 2>&1 | tee "$LOG"

  RC=${PIPESTATUS[0]}
  echo "Return code: $RC"
  echo "Saved log: $LOG"

  if [ $RC -ne 0 ]; then
    echo "FAILED: $DATASET $SAMPLER $RATIO"
    exit $RC
  fi
}

if [ "$MODE" = "smoke" ]; then
  echo "SMOKE TEST: chameleon, 1 seed, random vs IA at 25% budget."
  run_one chameleon random 0.25 1 smoke
  run_one chameleon ia 0.25 1 smoke
  python parse_budget_results.py
  echo "Smoke test completed."

elif [ "$MODE" = "main" ]; then
  echo "MAIN IA-BUDGET EXPERIMENTS"
  DATASETS="chameleon squirrel actor cora citeseer"
  RATIOS="0.25 0.10"
  SAMPLERS="random ia"
  NUMEXP=5

  for D in $DATASETS; do
    for R in $RATIOS; do
      for S in $SAMPLERS; do
        run_one "$D" "$S" "$R" "$NUMEXP" main
      done
    done
  done

  python parse_budget_results.py
  echo "Main IA-budget experiments completed."

elif [ "$MODE" = "ablation" ]; then
  echo "ABLATION EXPERIMENTS"
  DATASETS="chameleon squirrel cora"
  RATIO="0.10"
  SAMPLERS="spectral degree ia"
  NUMEXP=5

  for D in $DATASETS; do
    for S in $SAMPLERS; do
      run_one "$D" "$S" "$RATIO" "$NUMEXP" ablation
    done
  done

  python parse_budget_results.py
  echo "Ablation experiments completed."

else
  echo "Unknown mode: $MODE"
  echo "Use: smoke | main | ablation"
  exit 1
fi
