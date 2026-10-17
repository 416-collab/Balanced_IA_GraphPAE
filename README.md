# Balanced IA-GraphPAE

**Balanced Importance-Aware Positional Edge Selection for Budgeted Graph Positional Autoencoders**

This repository provides the PyTorch implementation of Balanced IA-GraphPAE, a graph representation learning approach that combines positional information and node-degree characteristics for budget-aware edge selection.

## Requirements

The implementation was tested with the following environment:

- Python 3.10.20
- PyTorch 2.11.0
- PyTorch Geometric 2.7.0
- CUDA 12.8
- NumPy, SciPy, scikit-learn and PyYAML

A CUDA-compatible GPU is recommended.

## Datasets

The implementation supports the following graph datasets:

| Dataset | Category |
|---|---|
| BlogCatalog | Social network |
| Chameleon | Wikipedia network |
| Squirrel | Wikipedia network |
| Actor | Film network |
| Cora | Citation network |
| CiteSeer | Citation network |
| Texas | WebKB |
| Cornell | WebKB |
| Wisconsin | WebKB |
| Penn94 | Social network |
| arXiv-year | Citation network |

Place the preprocessed dataset files in the `dataset/` directory. The expected file names correspond to the dataset configurations, including `arxiv-year-official.pt` for the official arXiv-year setup.

## Project Structure

```text
Balanced_IA_GraphPAE/
├── train_node.py
├── ia_utils.py
├── autoencoder.py
├── encoder.py
├── evaluation.py
├── preprocess.py
├── config/
├── scripts/
└── dataset/
```

## Running the Code

Install the required packages in a compatible Python environment and prepare the dataset files.

For example, to run the Cora experiment:

```bash
python train_node.py --dataset cora
```

Dataset-specific configurations are available in the `config/` directory.

## Implementation

Balanced IA-GraphPAE supports configurable edge-selection strategies and positional-edge budgets for graph representation learning. Training and evaluation parameters can be adjusted through the configuration files.

## Dataset Availability

Datasets are not included in this repository. Obtain the original datasets from their respective providers and preprocess them into the format expected by the implementation.
