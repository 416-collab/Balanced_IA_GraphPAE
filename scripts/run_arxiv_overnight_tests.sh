#!/usr/bin/env bash

cd ~/GraphPAE_Project/repo/GraphPAE-8a8cd33

source ~/miniconda3/etc/profile.d/conda.sh
conda activate graphpae

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=~/GraphPAE_Project/results_server/arxiv_overnight_tests
mkdir -p "$OUT"

run_test () {
  NAME="$1"
  LAYER="$2"
  DIM="$3"
  HEADS="$4"
  FREQ="$5"
  PEEDGES="$6"
  PELOSS="$7"
  GDROP="$8"
  EDROP="$9"

  echo "============================================================"
  echo "START TEST: $NAME"
  echo "layer=$LAYER dim=$DIM heads=$HEADS freq=$FREQ pe_edges=$PEEDGES pe_loss=$PELOSS gdrop=$GDROP edrop=$EDROP"
  echo "============================================================"

  python - <<PY
from pathlib import Path
import yaml

p = Path("config/arxiv-year.yaml")
c = yaml.safe_load(p.read_text())

c["device"] = 0
c["feat_dim"] = 1
c["epochs"] = 1000
c["epochs_eval"] = 2000
c["enc_gnn_layer"] = int("$LAYER")
c["embed_dim"] = int("$DIM")
c["heads"] = int("$HEADS")
c["max_freqs"] = int("$FREQ")
c["max_pe_edges"] = int("$PEEDGES")
c["masked_pe_loss"] = float("$PELOSS")
c["gnn_dropout"] = float("$GDROP")
c["gnn_edge_dropout"] = float("$EDROP")
c["mask_ratio"] = 0.5
c["replace_ratio"] = 0.0
c["masked_atom_loss"] = 1.0
c["atom_recon_type"] = "sce"
c["alpha_l"] = 2.0
c["noise_val"] = 0.01
c["optim"] = "adam"
c["init_lr"] = 0.001
c["weight_decay"] = 0.0
c["use_schedule"] = True
c["step_size"] = 50
c["gamma"] = 0.5
c["lr_eval"] = 0.01
c["wd_eval"] = 0.0001
c["gnn_activation"] = "elu"
c["task_head_activation"] = "gelu"
c["pos_activation"] = "gelu"

p.write_text(yaml.safe_dump(c, sort_keys=False))
print(p.read_text())
PY

  python -u train_node.py --dataset arxiv-year --num_exp 1 2>&1 | tee "$OUT/${NAME}.txt"

  echo "FINISHED TEST: $NAME"
  echo ""
  sleep 5
}

run_test arxiv_night_repeat_best_96d_100f_500k_pe005_drop03_edge05 1 96 1 100 500000 0.05 0.3 0.5
run_test arxiv_night_96d_100f_1M_pe005_drop03_edge05 1 96 1 100 1000000 0.05 0.3 0.5
run_test arxiv_night_128d_100f_500k_pe005_drop03_edge05 1 128 1 100 500000 0.05 0.3 0.5
run_test arxiv_night_128d_100f_1M_pe005_drop03_edge05 1 128 1 100 1000000 0.05 0.3 0.5
run_test arxiv_night_96d_100f_500k_pe010_drop03_edge05 1 96 1 100 500000 0.10 0.3 0.5
run_test arxiv_night_2layer_96d_100f_500k_pe005_drop03_edge05 2 96 1 100 500000 0.05 0.3 0.5
run_test arxiv_night_64d_2heads_100f_500k_pe005_drop03_edge05 1 64 2 100 500000 0.05 0.3 0.5

echo "============================================================"
echo "OVERNIGHT SUMMARY"
echo "============================================================"
grep -H "final result" "$OUT"/*.txt | tee "$OUT/arxiv_overnight_summary.txt"
