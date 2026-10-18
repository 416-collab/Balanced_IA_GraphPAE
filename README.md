# Balanced IA-GraphPAE — Reviewer staging package

Source: recorded final experiment snapshots of the October 2026 PRL revision, not the old public root defaults.

## Contents

- Nine core Python source files copied from the `ALL_REVISION` batch snapshot; `random_fixed` is included in that recorded source.
- Seven dataset YAML configurations sourced from the corresponding recorded final batch/config (see `SOURCE_PROVENANCE.md`).
- `evidence/revision_results/`: small, machine-readable run evidence (seed results, summary, effective configuration, hashes, environment); redundant per-run code snapshots, data tensors and model weights are intentionally omitted.
- `evidence/RUN_INVENTORY.csv` lists all included completed and exploratory run summaries; its presence does **not** mean every run was reported in the paper.

## Reproduction warning

This package is an **audit-ready staging candidate**, NOT an independently GPU-tested public release. The full processed datasets are not bundled. Paired-edge controls and Penn94 BF16 full-graph inference may depend on separate saved scripts/settings; inspect `scripts/` and archived run configuration before reproduction. Statistical claims and exact mapping of each manuscript table value still require an authoritative paper-to-run evidence index.

The original full server backup should be retained privately. No evidence in this package should be used to assert a result absent matching run records.
