#!/bin/bash
cd ~/GraphPAE_Project/repo/GraphPAE-8a8cd33
OUTDIR=~/GraphPAE_Project/results_server/FINAL_PAPER_RUN
mkdir -p "$OUTDIR"

DATASETS="blog chameleon squirrel actor texas cornell wisconsin cora citeseer"
declare -A STATUS

for NAME in $DATASETS; do
  echo ">>> Running $NAME ..."
  python train_node.py --dataset "$NAME" --num_exp 10 2>&1 | tee "$OUTDIR/${NAME}_final_10seed.txt"
  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    STATUS[$NAME]="OK"
  else
    STATUS[$NAME]="FAILED"
  fi
  echo ">>> Finished $NAME (${STATUS[$NAME]})"
  echo "---------------------------------------------"
done

echo ""
echo "=== RUN STATUS SUMMARY ==="
for NAME in $DATASETS; do
  echo "$NAME : ${STATUS[$NAME]}"
done

echo ""
echo "Building final summary table..."
python3 parse_final_results.py
