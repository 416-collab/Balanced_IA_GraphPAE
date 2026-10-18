# Raw-feature linear probe — Original experimental evidence

Source: `PRL_REVIEWER_PACKAGE.zip` provided in the conversation, original server path:
`revision_results/RAW_FEATURE_PROBE_20261018_120925_326896/`

**Verification:** all 35 original per-seed accuracy entries (7 datasets × seeds 0–4) recomputed using arithmetic mean and **population** standard deviation. Six datasets reproduce the manuscript (4).pdf Table 15 raw-feature column to two decimal places. Penn94 has a 0.01 percentage-point SD rounding difference: original five seeds give population SD 0.175034...%, conventionally 0.18% to two decimals, while the manuscript reports 0.17%. The original CSV rounds its SD to 0.175%.

| Dataset | Manuscript raw features (%, mean ± population SD) | Result |
|---|---|---|
| actor | 36.04 ± 0.89 | PASS |
| chameleon | 47.59 ± 2.11 | PASS |
| citeseer | 58.64 ± 0.05 | PASS |
| cora | 60.28 ± 0.04 | PASS |
| squirrel | 29.47 ± 1.28 | PASS |
| arxiv-year-official | 37.29 ± 0.20 | PASS |
| penn94 | 75.02 ± 0.17 | ROUNDING_0.01PP |

## Files

- `raw_probe_seed_results.json` — original unchanged 35 per-seed records; strongest numerical evidence.
- `raw_probe_summary.csv` — original unchanged server-generated summary; includes `paper_random` and `paper_ia` comparison columns which **must not** be used as authoritative manuscript comparators because some reflect earlier result batches.
- `TABLE15_RAW_FEATURE_VERIFICATION.csv` — independently recomputed from original seed JSON, exact Table 15 comparison.
- `SHA256.txt` — checksums of these three evidence files.

**Caution:** Penn94 75.02 ± 0.17% in the manuscript is not an exact two-decimal rounding of the seed-derived 75.0211 ± 0.17503%. This should be corrected or its rounding policy explicitly documented.

**Scope:** Evidence confirms recorded raw-feature linear-probe outputs and their agreement with the submitted manuscript. It does not independently rerun the GPU evaluation or independently validate dataset preprocessing. The archive was already provided; no new experiment or synthetic result was created.

To publish on GitHub, extract this archive under `evidence/raw_feature_probe/` and commit the new files.
