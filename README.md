# Balanced IA-GraphPAE — PRL revision

This repository contains the Balanced Importance-Aware GraphPAE source and
original experiment records for the Pattern Recognition Letters revision.

- **Implementation:** root Python files, `config/`, `scripts/`.
- **Review evidence:** `evidence/revision_results/`, including raw logs,
  seed-level results, effective configurations, and reviewer controls.
- **Integrity:** `SHA256_MANIFEST.json` records SHA-256 for every delivered file.
- **Provenance:** `REPRODUCIBILITY.md` explains the experiment archive.
- **Datasets:** not redistributed. Reproduce using the processed datasets
  and splits described in the article and experiment configuration records.

**Important:** historical `results/` entries from the earlier GitHub release
may use a different code snapshot, evaluation protocol or random seeds.
Do not treat them as the revised paper's results. The preserved revision
records are under `evidence/revision_results/`.

The positional-edge budget refers to selected directed **non-self** edges.
Self-loops are separately retained under the default protocol and are
removed in the dedicated self-loop ablation. Reported gains are
**dataset-dependent**, not universal.

See the article for full configuration details and dataset attribution.
