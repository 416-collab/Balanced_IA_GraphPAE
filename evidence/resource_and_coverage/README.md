# Evidence support: manuscript Tables 1 and 11

Source manuscript: `Balanced_Importance_Aware_Positional_Edge_Selection_for_Budgeted_Graph_Positional_Autoen__coders (4).pdf`, manuscript pages 6 and 15 (PDF pages 7 and 16). This bundle documents existing measurements and recorded server audit output. No new training, dataset loading, or experiment execution was performed to construct it.

## Table 1 — Five-seed pretraining allocated GPU memory

`TABLE01_ALL_SEED_MEMORY.csv` and the two original JSON copies contain all five peak allocated-memory readings for the **random reference** on each large graph. `TABLE01_RUN_MAPPING.md` maps them to original archived paths and recomputes the max across seeds. Manuscript arXiv-year 18.880 GiB and Penn94 8.619 GiB both match when rounded to 3 decimals. These are training-loop allocation peaks, not inference, full device usage, or evidence of a maximal feasible budget.

## Table 11 — Node coverage and reconstruction pairs

`TABLE11_ORIGINAL_SERVER_TRANSCRIPT_EXCERPT.txt` preserves the exact earlier server-session outputs for **five datasets, two budgets, and two selectors**, including selected directed non-self edges, unique reconstructed pairs, covered nodes, and an illustrative loss-edge count. `TABLE11_RECORDED_AUDIT_CODE_EXCERPT.txt` preserves the available code excerpt. The audit used **fixed seed 0**, and did not train models. `TABLE11_AUDIT_10_AND_25_PERCENT.csv` is a transcription of the original captured output, not simulated selection. `TABLE11_MANUSCRIPT_MATCH.csv` checks all five 10% rows against the exact manuscript's Table 11; all match at displayed precision. Coverage = touched node count / all nodes; loop share = N/(N+B), and unique unordered reconstruction pairs = symmetrized_nonself_entries/2. Table 11 is an edge-selection audit, **not a five-seed accuracy analysis**.

### Important limitations and provenance

- The original Table 11 outputs were captured in the previously uploaded **server terminal session text**, rather than preserved as a machine-readable `.json` generated on the server. This package copies that output verbatim and provides a structured CSV with formulas and match checks. It does **not** claim to be a fresh independent server rerun or a signed machine-generated log.
- The input graph tensors and complete processing code were **not rerun** for these new evidence documents. To independently reproduce, use the matching `ia_utils.py`, data files/splits, and configurations from the original run, and audit fixed seed 0.
- The separately recorded **Penn94 BF16 full-graph inference** peak of 17.266 GiB is excerpted as supplementary context. It must not be confused with the Penn94 **8.619 GiB pretraining** peak in Table 1.
- All path references in the mapping documents identify source files inside the previously supplied ZIP archive; these names can be resolved against GitHub's `evidence/revision_results/` tree.

## Files

- `TABLE01_RUN_MAPPING.md`: source paths and five-seed peak reconciliation
- `TABLE01_ALL_SEED_MEMORY.csv`: each original seed measurement with source path
- `TABLE01_*_original_seed_results.json`: byte-for-byte copies of original five-seed JSON
- `TABLE11_ORIGINAL_SERVER_TRANSCRIPT_EXCERPT.txt`: original terminal audit printed measurements
- `TABLE11_RECORDED_AUDIT_CODE_EXCERPT.txt`: original terminal script excerpt
- `TABLE11_AUDIT_10_AND_25_PERCENT.csv`: transcript-derived structured table
- `TABLE11_MANUSCRIPT_MATCH.csv`: 10% manuscript-table match
- `PENN94_BF16_INFERENCE_CHECK_TRANSCRIPT.txt`: separate Penn94 BF16 inference-check excerpt
- `SHA256_MANIFEST.txt`: checksums over files in this evidence folder
