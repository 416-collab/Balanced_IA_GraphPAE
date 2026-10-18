# Balanced IA-GraphPAE

**Balanced Importance-Aware Positional Edge Selection for Budgeted Graph Positional Autoencoders**

Research code, experimental configurations, and recorded evidence accompanying the October 2026 manuscript revision.

## Overview

This repository investigates importance-aware positional edge selection for graph positional autoencoders under limited positional-edge budgets.

The proposed **Balanced Importance-Aware (IA)** strategy combines spectral and degree-based importance estimates with distance-rank binning to select positional edges while balancing representation across distance ranges.

The study evaluates whether importance-aware selection provides benefits over random and other edge-selection strategies across multiple graph datasets and budget settings.

## Main Contributions

- **Budget-aware positional edge selection:** Selects a limited number of edges for positional reconstruction.
- **Balanced importance-aware sampling:** Combines spectral and degree information with distance-rank bins.
- **Controlled comparisons:** Evaluates refreshed random, fixed random, spectral, and degree-based selection.
- **Component ablations:** Examines the contributions of importance scoring and bin-based balancing.
- **Additional evaluation protocols:** Includes paired-edge controls, self-loop analysis, budget sensitivity, and large-graph experiments.

## Evaluated Datasets

Seven datasets are included in the experimental evaluation:

| Dataset | Evaluation scope |
|---|---|
| Actor | Small-graph comparisons |
| Chameleon | Comparisons, ablations, budget sensitivity, paired-edge controls |
| CiteSeer | Small-graph comparisons |
| Cora | Comparisons, ablations, paired-edge controls |
| Squirrel | Comparisons, ablations, budget sensitivity |
| arXiv-year | Large-graph evaluation |
| Penn94 | Large-graph evaluation, BF16 inference |

The large-graph experiments use bounded positional-edge selection. The recorded Penn94 experiments include a 200,000-edge cap; arXiv-year experiments include a 500,000-edge cap.

## Repository Structure

```text
Balanced_IA_GraphPAE/
├── train_node.py
├── autoencoder.py
├── encoder.py
├── conv.py
├── pos_enc.py
├── ia_utils.py
├── evaluation.py
├── preprocess.py
├── utils.py
├── config/
├── scripts/
├── revision_tools/
├── evidence/
│   ├── RUN_INVENTORY.csv
│   └── revision_results/
├── SOURCE_PROVENANCE.md
├── SHA256_MANIFEST.txt
└── README.md
```

### Source Code and Configurations

The core Python modules and seven dataset-specific YAML configurations are taken from recorded October 2026 revision snapshots.

`SOURCE_PROVENANCE.md` documents the source selection and configuration provenance. Individual recorded experiments may contain settings that differ from repository-level defaults.

### Experimental Evidence

The `evidence/revision_results/` directory contains 110 recorded experiment summaries, covering main comparisons, auxiliary controls, ablations, and exploratory experiments.

Available records include, where captured:

- `summary.json` — aggregate accuracy statistics.
- `seed_results.json` — individual seed-level results.
- `effective_config.json` — effective experiment settings.
- `environment.json` — recorded software and hardware environment.
- `code_hashes.json` — source-code provenance hashes.

The [`evidence/RUN_INVENTORY.csv`](evidence/RUN_INVENTORY.csv) file provides an inventory of the archived experiments.

**Important:** The inventory includes exploratory experiments and additional controls. Not every recorded run corresponds to a reported manuscript result.

## Experimental Protocol

The recorded evaluations generally use five runs, with aggregate accuracy reported as mean and standard deviation.

The experimental comparisons cover multiple selection methods and positional-edge budgets. In the primary directed-edge protocol, selected non-self directed edges consume the positional-edge budget; self-loops are handled separately.

Paired-edge controls are reported as a distinct protocol to help assess the effect of edge-direction accounting.

For exact run-specific settings, consult each experiment's `effective_config.json`.

## Reproducibility and Limitations

This repository provides recorded research artifacts and a compact experimental evidence archive.

The following limitations apply:

1. The repository has not been independently verified through a fresh end-to-end GPU reproduction.
2. Full processed datasets, model checkpoints, and redundant per-run code snapshots are not bundled.
3. Some specialized experiments may require additional run-specific scripts or settings beyond the repository defaults.
4. Exact correspondence between a manuscript table entry and its source run should be established using the recorded experiment configuration and seed results.
5. The archived experimental evidence should not be interpreted as proof of results that lack corresponding source records.

The original complete experiment archive should be retained separately.

## Evidence Integrity

The repository includes `SHA256_MANIFEST.txt` for the packaged files.

The archived five-seed experiment summaries have been checked for internal numerical consistency against their saved seed-level accuracy records. This is an evidence-consistency check, not an independent re-execution of the experiments.

## Research Use

Researchers may inspect the implementation, evaluate the selection mechanisms, and use the provided experiment records to understand the reported methodology.

For formal reproduction, use the dataset-specific configurations and check the original effective settings of the experiments being replicated.

## Citation

If you use this repository, please cite the associated manuscript:

**Balanced Importance-Aware Positional Edge Selection for Budgeted Graph Positional Autoencoders**

Muhammad Shaban Tabassam, Wang Zhixiao, Shaopeng Che, and Haochen Bai.

*Manuscript under review/revision, 2026.*

A final bibliographic citation will be added upon publication.

## Contact

For questions about the implementation or experimental evidence, please open a GitHub issue in this repository.
