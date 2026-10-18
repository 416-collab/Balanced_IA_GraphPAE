#!/usr/bin/env python3
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path("/home/insuper/GraphPAE_Project/PRL_Revision_20261014_004139_949694")
if len(sys.argv) != 2:
    raise SystemExit("usage: python revision_tools/collect.py NAME")
batches = sorted(p for p in (ROOT / "revision_results").glob(sys.argv[1] + "_*")
                 if (p / "JOBS.json").is_file())
if not batches:
    raise SystemExit("No batch found for " + sys.argv[1])
batch = batches[-1]
rows, problems = [], []
for job in json.loads((batch / "JOBS.json").read_text()):
    log = batch / (job["tag"] + ".log")
    if not log.exists():
        problems.append(job["tag"] + ": not started")
        continue
    dirs = [line.split("RUN DIRECTORY:", 1)[1].strip()
            for line in log.read_text(errors="replace").splitlines()
            if "RUN DIRECTORY:" in line]
    if len(dirs) != 1:
        problems.append(job["tag"] + ": run directory missing or ambiguous")
        continue
    run = Path(dirs[0])
    try:
        summary = json.loads((run / "summary.json").read_text())
        seeds = sorted(json.loads((run / "seed_results.json").read_text()),
                       key=lambda row: row["seed"])
        cfg = json.loads((run / "effective_config.json").read_text())
    except FileNotFoundError:
        problems.append(job["tag"] + ": incomplete (still running or failed)")
        continue
    values = np.array([row["accuracy"] for row in seeds], dtype=float)
    checks = {
        "seeds 0-4": [row["seed"] for row in seeds] == list(range(5)),
        "accuracies": bool(np.allclose(values, summary["accuracies"])),
        "mean": abs(values.mean() - summary["mean"]) < 1e-10,
        "std": abs(values.std() - summary["std_population"]) < 1e-10,
        "sampler": cfg.get("pe_sampler") == job["sampler"],
        "budget": abs(float(cfg.get("pe_budget_ratio")) - job["budget"]) < 1e-9,
        "self-loops kept": cfg.get("pe_keep_self_loops") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        problems.append(job["tag"] + ": CHECK FAILED " + ", ".join(failed))
        continue
    secs = [r["pretrain_seconds"] for r in seeds if r.get("pretrain_seconds") is not None]
    peaks = [r["pretrain_peak_allocated_gib"] for r in seeds
             if r.get("pretrain_peak_allocated_gib") is not None]
    rows.append({
        "dataset": job["dataset"], "sampler": job["sampler"],
        "budget": job["budget"],
        "mean_percent": round(values.mean() * 100, 3),
        "sd_percent": round(values.std() * 100, 3),
        "seed_accuracies_percent": " ".join(f"{v * 100:.2f}" for v in values),
        "mean_pretrain_seconds": round(float(np.mean(secs)), 2) if secs else "",
        "max_pretrain_peak_gib": round(max(peaks), 3) if peaks else "",
        "run_directory": str(run),
    })

print("BATCH:", batch)
print(f"{'dataset':<10} {'sampler':<14} {'budget':>6}   accuracy (mean ± pop. SD)")
for row in rows:
    print(f"{row['dataset']:<10} {row['sampler']:<14} {row['budget']:>6.2f}   "
          f"{row['mean_percent']:.2f} ± {row['sd_percent']:.2f}   "
          f"[{row['seed_accuracies_percent']}]")
if rows:
    out = batch / "RESULTS_SUMMARY.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nCSV:", out)
print("\nARITHMETIC/CONFIG CHECKS:", "PASS" if not problems else "SEE BELOW")
for problem in problems:
    print("  -", problem)
