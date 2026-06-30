
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from evaluation import node_evaluation

def select_mask(mask, seed):
    if mask.dim() == 1:
        return mask
    elif mask.dim() == 2:
        return mask[:, seed % mask.size(1)]
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.dim()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = torch.load(ROOT / "dataset/arxiv-year.pt", weights_only=False)

x = data.x.float().to(device)
y = data.y.to(device)

print("x:", x.shape)
print("y:", y.shape, y.min().item(), y.max().item())
print("train_mask:", data.train_mask.shape)
print("val_mask:", data.val_mask.shape)
print("test_mask:", data.test_mask.shape)

results = []

for seed in range(10):
    train_mask = select_mask(data.train_mask, seed).to(device)
    val_mask = select_mask(data.val_mask, seed).to(device)
    test_mask = select_mask(data.test_mask, seed).to(device)

    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]

    acc, pred = node_evaluation(
        emb=x,
        y=y,
        train_idx=train_idx,
        valid_idx=val_idx,
        test_idx=test_idx,
        epochs=2000,
        lr=0.01,
        weight_decay=0.0001,
    )

    print(f"seed {seed}: {acc}")
    results.append(float(acc))

print(results)
print(f"RAW FEATURE FINAL: {np.mean(results):.5f}±{np.std(results):.5f}")
