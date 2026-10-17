# PRL Revision — evidence provenance

Pack generated (UTC): 2026-10-17T06:14:51.194035+00:00
Original server project: `/home/insuper/GraphPAE_Project/PRL_Revision_20261014_004139_949694`
Paired-edge control batch: `PAIRED_CONTROL_20261015_005810_312367`
Paired-edge completed log files checked: 8
Seed-level result/summary pairs validated: 53

## Interpretation

`evidence/revision_results/` is the preserved experiment audit trail.
`evidence/historical_results_server/` contains older GraphPAE experiments when available;
those records must NOT be conflated with the controlled PRL-revision runs.
It contains main runs, ablations, budget sensitivity, large graphs,
self-loop tests, and paired-edge controls where recorded. This archive
also contains exploratory runs: **not every result is reported in the paper**.
Compare the paper to the corresponding raw logs and effective configurations.
The selector's non-self directed edge budget is distinct from active encoder
entries including self-loops; their loss contribution is separately defined.

## Verification and limitations

Code was syntax-checked, paired-control final logs were checked, selected
seed-result means and population SDs were recomputed, and archived bytes
were checked with SHA-256. This packaging process does **not** rerun
experiments or establish that every manuscript number matches a log.
Dataset tensors, checkpoints, server machine files, and secrets are excluded.
The original working source tree was not changed by the packager.
