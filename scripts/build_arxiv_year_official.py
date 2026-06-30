
import time
import gc
from pathlib import Path

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

from torch_geometric.datasets import LINKXDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset"
OUT_NAME = "arxiv-year-official"
K = 100


def compute_sparse_lap_pe(edge_index, num_nodes, k, cache_prefix):
    e_path = DATASET_ROOT / f"{cache_prefix}_e_k{k}.pt"
    u_path = DATASET_ROOT / f"{cache_prefix}_u_k{k}.pt"

    if e_path.exists() and u_path.exists():
        print("Loading cached PE:")
        print(e_path)
        print(u_path)
        return torch.load(e_path, weights_only=False), torch.load(u_path, weights_only=False)

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(edge_index)

    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()

    adj = sp.coo_matrix(
        (np.ones(row.shape[0], dtype=np.float32), (row, col)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsr()

    adj.setdiag(0)
    adj.eliminate_zeros()

    print("Adjacency:", adj.shape, "nnz:", adj.nnz)

    L = sparse_laplacian(adj, normed=True).astype(np.float32).tocsr()

    print(f"Computing {k} smallest eigenvectors...")
    start = time.time()

    vals, vecs = eigsh(L, k=k, which="SA", tol=1e-3, maxiter=10000)

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


print("Loading LINKXDataset arxiv-year...")
ds = LINKXDataset(root=str(DATASET_ROOT / "LINKX"), name="arxiv-year")
data = ds[0]

print(data)
print("x:", data.x.shape)
print("edge_index:", data.edge_index.shape)
print("y:", data.y.shape, data.y.min().item(), data.y.max().item())

for name in ["train_mask", "val_mask", "test_mask"]:
    if hasattr(data, name):
        print(name, getattr(data, name).shape)
    else:
        print("Missing:", name)

x = data.x.float()
edge_index = to_undirected(data.edge_index.long(), num_nodes=x.size(0))
y = data.y.long().view(-1)

e, u = compute_sparse_lap_pe(
    edge_index=edge_index,
    num_nodes=x.size(0),
    k=K,
    cache_prefix=OUT_NAME,
)

out = Data(x=x, edge_index=edge_index, y=y)
out.train_mask = data.train_mask
out.val_mask = data.val_mask
out.test_mask = data.test_mask
out.e = e
out.u = u

save_path = DATASET_ROOT / f"{OUT_NAME}.pt"
torch.save(out, save_path)

print("Saved:", save_path)
print("Final:")
print(out)
print("train_mask:", out.train_mask.shape)
print("val_mask:", out.val_mask.shape)
print("test_mask:", out.test_mask.shape)
