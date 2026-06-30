
import os
import time
import gc
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from ogb.nodeproppred import NodePropPredDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def even_quantile_labels(vals, nclasses=5):
    vals = np.asarray(vals).reshape(-1)
    quantiles = np.percentile(vals, np.linspace(0, 100, nclasses + 1))
    labels = np.zeros(vals.shape[0], dtype=np.int64)

    for i in range(nclasses):
        if i == 0:
            mask = vals <= quantiles[i + 1]
        elif i == nclasses - 1:
            mask = vals > quantiles[i]
        else:
            mask = (vals > quantiles[i]) & (vals <= quantiles[i + 1])
        labels[mask] = i

    return labels


def make_random_masks(y, num_splits=10, train_prop=0.5, val_prop=0.25, seed_base=0):
    y_cpu = y.cpu()
    valid_idx = torch.where(y_cpu >= 0)[0]
    n = valid_idx.numel()

    train_masks, val_masks, test_masks = [], [], []

    for s in range(num_splits):
        g = torch.Generator()
        g.manual_seed(seed_base + s)

        perm = valid_idx[torch.randperm(n, generator=g)]

        n_train = int(train_prop * n)
        n_val = int(val_prop * n)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

        train_mask = torch.zeros(y_cpu.size(0), dtype=torch.bool)
        val_mask = torch.zeros(y_cpu.size(0), dtype=torch.bool)
        test_mask = torch.zeros(y_cpu.size(0), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    return (
        torch.stack(train_masks, dim=1),
        torch.stack(val_masks, dim=1),
        torch.stack(test_masks, dim=1),
    )


def compute_sparse_lap_pe(edge_index, num_nodes, k, cache_prefix):
    e_path = DATASET_ROOT / f"{cache_prefix}_e_k{k}.pt"
    u_path = DATASET_ROOT / f"{cache_prefix}_u_k{k}.pt"

    if e_path.exists() and u_path.exists():
        print("Loading cached PE:")
        print(e_path)
        print(u_path)
        e = torch.load(e_path, weights_only=False)
        u = torch.load(u_path, weights_only=False)
        return e, u

    print("Converting graph to undirected and removing self-loops...")
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)

    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()

    print("Building sparse adjacency...")
    adj = sp.coo_matrix(
        (np.ones(row.shape[0], dtype=np.float32), (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsr()

    adj.setdiag(0)
    adj.eliminate_zeros()

    print("Adjacency:", adj.shape, "nnz:", adj.nnz)

    print("Building normalized sparse Laplacian...")
    L = sparse_laplacian(adj, normed=True).astype(np.float32).tocsr()

    print(f"Computing {k} smallest eigenvectors with scipy eigsh...")
    print("This may take time. Do not close the tmux session.")
    start = time.time()

    vals, vecs = eigsh(
        L,
        k=k,
        which="SA",
        tol=1e-3,
        maxiter=10000
    )

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    e = torch.tensor(vals, dtype=torch.float32)
    u = torch.tensor(vecs, dtype=torch.float32)

    torch.save(e, e_path)
    torch.save(u, u_path)

    print("PE saved:")
    print(e_path)
    print(u_path)
    print("PE time minutes:", round((time.time() - start) / 60, 2))

    del adj, L, vals, vecs
    gc.collect()

    return e, u


def preprocess_arxiv_year(k=100):
    print("=" * 80)
    print("Preprocessing arxiv-year")
    print("=" * 80)

    save_path = DATASET_ROOT / "arxiv-year.pt"
    if save_path.exists():
        print("arxiv-year.pt already exists:")
        print(save_path)
        return save_path

    ogb_root = DATASET_ROOT / "OGB"
    dataset = NodePropPredDataset(name="ogbn-arxiv", root=str(ogb_root))
    graph = dataset.graph

    x = torch.tensor(graph["node_feat"], dtype=torch.float32)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    years = graph["node_year"].reshape(-1)

    y_np = even_quantile_labels(years, nclasses=5)
    y = torch.tensor(y_np, dtype=torch.long)

    num_nodes = x.size(0)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    print("Nodes:", num_nodes)
    print("Edges after undirected:", edge_index.size(1))
    print("Features:", x.size(1))
    print("Classes:", int(y.max().item() + 1))

    train_mask, val_mask, test_mask = make_random_masks(
        y,
        num_splits=10,
        train_prop=0.5,
        val_prop=0.25,
        seed_base=0
    )

    e, u = compute_sparse_lap_pe(
        edge_index=edge_index,
        num_nodes=num_nodes,
        k=k,
        cache_prefix="arxiv-year"
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.e = e
    data.u = u

    torch.save(data, save_path)
    print("Saved:", save_path)
    return save_path


if __name__ == "__main__":
    preprocess_arxiv_year(k=100)
