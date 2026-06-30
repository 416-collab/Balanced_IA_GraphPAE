import re
import csv
from pathlib import Path
import numpy as np

ROOT = Path.home() / "GraphPAE_Project/results_server/FINAL_ALL_DATASETS"
LOG_DIR = ROOT / "logs"
OUT_DIR = Path.home() / "GraphPAE_Project/results_server/FINAL_BEST_SINGLE_RUN_ASSIGNMENT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for log_path in sorted(LOG_DIR.glob("*.txt")):
    text = log_path.read_text(errors="ignore")

    dataset_match = re.search(r"dataset='([^']+)'", text)
    num_exp_match = re.search(r"num_exp=([0-9]+)", text)

    if not dataset_match:
        continue

    dataset = dataset_match.group(1)
    num_exp = int(num_exp_match.group(1)) if num_exp_match else None

    accs = [float(x) for x in re.findall(r"Epoch\s+1000,\s+ACC:\s+([0-9.]+)", text)]

    final_match = re.search(r"final result:\s*([0-9.]+)±([0-9.]+)", text)
    if final_match:
        mean = float(final_match.group(1))
        std = float(final_match.group(2))
    else:
        mean = float(np.mean(accs)) if accs else None
        std = float(np.std(accs)) if accs else None

    if not accs:
        continue

    best = max(accs)
    best_seed = accs.index(best)

    rows.append({
        "dataset": dataset,
        "runs": num_exp,
        "best_seed_index": best_seed,
        "best_single_run_acc": best,
        "best_single_run_percent": best * 100,
        "mean_acc": mean,
        "std_acc": std,
        "mean_percent": mean * 100 if mean is not None else None,
        "std_percent": std * 100 if std is not None else None,
        "all_accs": accs,
        "log_file": str(log_path),
    })

# CSV
csv_path = OUT_DIR / "BEST_SINGLE_RUN_ASSIGNMENT_RESULTS.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset", "runs", "best_seed_index",
            "best_single_run_acc", "best_single_run_percent",
            "mean_acc", "std_acc", "mean_percent", "std_percent",
            "all_accs", "log_file"
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

# Markdown table
md_path = OUT_DIR / "BEST_SINGLE_RUN_ASSIGNMENT_RESULTS.md"
lines = []
lines.append("# Best Single-Run Assignment Results")
lines.append("")
lines.append("Important: Raw logs were not edited. This table reports the highest single-run accuracy from each saved run file.")
lines.append("")
lines.append("| Dataset | Runs | Best single-run (%) | Best seed index | Mean ± std (%) |")
lines.append("|---|---:|---:|---:|---:|")

for r in rows:
    lines.append(
        f"| {r['dataset']} | {r['runs']} | "
        f"{r['best_single_run_percent']:.2f} | "
        f"{r['best_seed_index']} | "
        f"{r['mean_percent']:.2f} ± {r['std_percent']:.2f} |"
    )

md_path.write_text("\n".join(lines))

# Per-dataset clean text files
for r in rows:
    out = OUT_DIR / f"{r['dataset']}_BEST_SINGLE_RUN.txt"
    out.write_text(
        f"Dataset: {r['dataset']}\n"
        f"Runs: {r['runs']}\n"
        f"Best seed index: {r['best_seed_index']}\n"
        f"Best single-run ACC: {r['best_single_run_acc']:.8f}\n"
        f"Best single-run percentage: {r['best_single_run_percent']:.2f}%\n"
        f"Original mean ± std ACC: {r['mean_acc']:.8f} ± {r['std_acc']:.8f}\n"
        f"Original mean ± std percentage: {r['mean_percent']:.2f} ± {r['std_percent']:.2f}%\n"
        f"All ACC values:\n{r['all_accs']}\n"
        f"Original log file:\n{r['log_file']}\n"
    )

print("Saved:", csv_path)
print("Saved:", md_path)
print()
print(md_path.read_text())
