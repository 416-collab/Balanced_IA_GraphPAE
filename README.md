# Balanced IA-GraphPAE

Clean implementation package for **Balanced Importance-Aware Positional Edge Selection for Budgeted Graph Positional Autoencoders**.

This README documents only the environment, requirements, dataset sources, and basic usage. It intentionally does **not** report experimental results.

## 1. Environment

Recommended Conda environment:

```bash
conda create -n graphpae python=3.10 -y
conda activate graphpae
```

The project was prepared for a Linux server environment with:

- Python 3.10
- CUDA-enabled NVIDIA GPU
- PyTorch
- PyTorch Geometric
- Conda environment name: `graphpae`

A GPU is strongly recommended because GraphPAE-style positional autoencoding and PE-edge experiments are memory intensive.

## 2. Python requirements

Install a PyTorch build compatible with your CUDA version first. Then install the remaining packages.

Core packages:

```text
torch
torch-geometric
numpy
scipy
scikit-learn
PyYAML
tqdm
networkx
pandas
ogb
```

Example installation pattern:

```bash
conda activate graphpae

# Install PyTorch according to your CUDA version from the official PyTorch selector.
# Then install the remaining packages:
pip install torch-geometric numpy scipy scikit-learn PyYAML tqdm networkx pandas ogb
```

If PyTorch Geometric requires additional CUDA wheels for your PyTorch/CUDA version, follow the official PyTorch Geometric installation instructions.

## 3. Main files

```text
ia_utils.py
train_node.py
set_budget_config.py
run_random_budget.sh
run_balanced_ia_budget.sh
parse_final_results.py
config/
results/
```

Key implementation files:

- `ia_utils.py` — Balanced importance-aware positional edge selector.
- `train_node.py` — Training script with active PE-edge selection and active PE-value recomputation.
- `set_budget_config.py` — Helper script for setting dataset, sampler, and budget ratio.
- `run_random_budget.sh` — Runs random PE-edge budget baseline.
- `run_balanced_ia_budget.sh` — Runs Balanced IA PE-edge budget experiments.
- `parse_final_results.py` — Parses saved log files.

## 4. Dataset sources

This clean repository does **not** include raw or processed dataset files.

The project expects processed PyTorch files in:

```text
./dataset/{dataset}.pt
```

Required processed files:

```text
./dataset/chameleon.pt
./dataset/squirrel.pt
./dataset/actor.pt
./dataset/cora.pt
./dataset/citeseer.pt
```

Dataset source mapping:

| Dataset | Source family | PyTorch Geometric dataset class |
|---|---|---|
| Cora | Citation network dataset | `torch_geometric.datasets.Planetoid` |
| CiteSeer | Citation network dataset | `torch_geometric.datasets.Planetoid` |
| Chameleon | Wikipedia network dataset | `torch_geometric.datasets.WikipediaNetwork` |
| Squirrel | Wikipedia network dataset | `torch_geometric.datasets.WikipediaNetwork` |
| Actor | Actor-only Geom-GCN dataset | `torch_geometric.datasets.Actor` |

Official PyTorch Geometric documentation:

- Planetoid datasets: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html
- WikipediaNetwork datasets: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html
- Actor dataset: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Actor.html

## 5. Dataset preparation

Before running experiments, create a `dataset/` folder inside the repository:

```bash
mkdir -p dataset
```

Then copy or generate the processed dataset files:

```text
dataset/chameleon.pt
dataset/squirrel.pt
dataset/actor.pt
dataset/cora.pt
dataset/citeseer.pt
```

The repository assumes that the original GraphPAE data processing pipeline has already converted each dataset into the expected `.pt` format.

## 6. Budgeted PE-edge selection

The budget is applied only to **non-self positional edges**.

Self-loops are always retained to preserve GAT edge-attribute alignment.

The selected active positional edge set is used for both:

1. positional message passing, and
2. positional reconstruction.

The active PE values are recomputed from the selected active edge set:

```python
active_edge_index = selected_positional_edges
active_PE = recomputed_PE_values_for_active_edge_index
graph_auto_encoder(x, active_edge_index, u, active_PE, active_edge_index)
```

## 7. Running experiments

Run random PE-edge budget baseline:

```bash
conda activate graphpae
bash run_random_budget.sh
```

Run Balanced IA PE-edge budget experiments:

```bash
conda activate graphpae
bash run_balanced_ia_budget.sh
```

Parse saved logs:

```bash
python parse_final_results.py
```

## 8. Notes

- The clean repository intentionally excludes raw datasets, processed `.pt` datasets, checkpoints, cache files, failed logs, and server-specific temporary files.
- To rerun experiments, the processed dataset files must be placed under `./dataset/`.
- Configuration files are stored in `config/`.
- The default configuration is neutral; use `set_budget_config.py` or the provided run scripts to set the sampler and PE-edge budget before running experiments.
