#!/usr/bin/env python3
"""Raw-feature linear probe: same splits, seeds and evaluation code as GraphPAE runs."""
from pathlib import Path
from datetime import datetime
import csv
import gc
import json
import subprocess
import sys

import numpy as np
import torch

ROOT = Path("/home/insuper/GraphPAE_Project/PRL_Revision_20261014_004139_949694")
sys.path.insert(0, str(ROOT))
from evaluation import node_evaluation  # noqa: E402
from utils import set_random_seed  # noqa: E402

# name, file, classifier weight decay (lr 0.01 and 2000 epochs for all)
DATASETS = [
    ("cora", "cora.pt", 0.0001),
    ("citeseer", "citeseer.pt", 0.0001),
    ("chameleon", "chameleon.pt", 0.0),
    ("squirrel", "squirrel.pt", 0.0),
    ("actor", "actor.pt", 0.0001),
    ("arxiv-year-official", "arxiv-year-official.pt", 0.0001),
    ("penn94", "penn94_full_features_backup.pt", 0.0001),
]

# Revised-paper GraphPAE results: (random, Balanced IA)
PAPER = {
    "cora": (81.96, 79.56),
    "citeseer": (67.64, 66.96),
    "chameleon": (63.68, 68.29),
    "squirrel": (35.72, 39.54),
    "actor": (30.34, 25.26),
    "arxiv-year-official": (40.09, 39.80),
    "penn94": (60.14, 56.11),
}

assert torch.cuda.is_available(), "CUDA unavailable"

busy = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
    text=True
).strip()

if busy:
    print("NOTE: other GPU jobs are running; probe accuracy is unaffected.")

device = torch.device("cuda:0")

out = ROOT / "revision_results" / (
    f"RAW_FEATURE_PROBE_{datetime.now():%Y%m%d_%H%M%S_%f}"
)
out.mkdir(parents=True)

records, summary = [], []

for name, filename, wd in DATASETS:
    data = torch.load(
        ROOT / "dataset" / filename,
        map_location="cpu",
        weights_only=False
    )

    x = data.x.float().to(device)
    y = data.y.reshape(-1).long().to(device)
    accs, majs = [], []

    for seed in range(5):
        index = []

        for key in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(data, key)
            mask = mask if mask.dim() == 1 else mask[:, seed % mask.size(1)]
            index.append(torch.where(mask.bool())[0].to(device))

        train_idx, val_idx, test_idx = index

        assert int((y[train_idx] < 0).sum()) == 0
        assert int((y[test_idx] < 0).sum()) == 0

        set_random_seed(10000 + seed)

        acc, _ = node_evaluation(
            emb=x,
            y=y,
            train_idx=train_idx,
            valid_idx=val_idx,
            test_idx=test_idx,
            epochs=2000,
            lr=0.01,
            weight_decay=wd
        )

        majority = int(torch.bincount(y[train_idx]).argmax())
        maj = float((y[test_idx] == majority).float().mean())

        accs.append(float(acc))
        majs.append(maj)

        records.append({
            "dataset": name,
            "seed": seed,
            "raw_probe_accuracy": float(acc),
            "majority_class_accuracy": maj,
            "features": int(x.size(1))
        })

        print(
            f"{name:<20} seed {seed}: "
            f"raw probe {float(acc) * 100:6.2f}% | "
            f"majority class {maj * 100:6.2f}%",
            flush=True
        )

    a = np.array(accs) * 100
    m = np.array(majs) * 100

    summary.append({
        "dataset": name,
        "features": int(x.size(1)),
        "raw_mean": round(a.mean(), 3),
        "raw_sd": round(a.std(), 3),
        "majority_mean": round(m.mean(), 3),
        "paper_random": PAPER[name][0],
        "paper_ia": PAPER[name][1]
    })

    del data, x, y
    gc.collect()
    torch.cuda.empty_cache()

(out / "raw_probe_seed_results.json").write_text(
    json.dumps(records, indent=2)
)

with (out / "raw_probe_summary.csv").open("x", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(
    "\nDATASET              RAW PROBE (mean ± SD)   "
    "MAJORITY   PAPER RANDOM   PAPER IA   GraphPAE > raw?"
)

for row in summary:
    best = max(row["paper_random"], row["paper_ia"])
    flag = "yes" if best > row["raw_mean"] else "NO"

    print(
        f"{row['dataset']:<20} "
        f"{row['raw_mean']:6.2f} ± {row['raw_sd']:5.2f}"
        f"          {row['majority_mean']:6.2f}"
        f"     {row['paper_random']:6.2f}"
        f"        {row['paper_ia']:6.2f}     {flag}"
    )

print("\nSAVED:", out)
