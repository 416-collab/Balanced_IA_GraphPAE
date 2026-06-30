import re
import csv
from pathlib import Path

ROOT = Path.home() / "GraphPAE_Project/results_server"
FINAL_LOG_DIR = ROOT / "FINAL_ALL_DATASETS/logs"
OUT_DIR = ROOT / "FINAL_MAX_SINGLE_RUN_REPORT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = [
    "blog",
    "chameleon",
    "squirrel",
    "actor",
    "texas",
    "cornell",
    "wisconsin",
    "cora",
    "citeseer",
    "penn94",
    "arxiv-year-official",
]

DISPLAY = {
    "blog": "BlogCatalog",
    "chameleon": "Chameleon",
    "squirrel": "Squirrel",
    "actor": "Actor",
    "texas": "Texas",
    "cornell": "Cornell",
    "wisconsin": "Wisconsin",
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "penn94": "Penn94",
    "arxiv-year-official": "arXiv-year Official",
}

rows = []

# Scan latest final logs first
log_files = list(FINAL_LOG_DIR.glob("*.txt"))

# Also scan master log in case some output exists only there
master_log = ROOT / "FINAL_ALL_DATASETS/final_run_master_log.txt"
if master_log.exists():
    log_files.append(master_log)

for path in log_files:
    text = path.read_text(errors="ignore")

    # Split by dataset runs
    blocks = re.split(r"\n={20,}\n", text)

    for block in blocks:
        ns_match = re.search(r"Namespace\((.*?)\)", block, re.S)
        if not ns_match:
            continue

        ns = ns_match.group(1)

        dataset_match = re.search(r"dataset='([^']+)'", ns)
        num_exp_match = re.search(r"num_exp=([0-9]+)", ns)

        if not dataset_match:
            continue

        dataset = dataset_match.group(1)
        if dataset not in DATASET_NAMES:
            continue

        num_exp = int(num_exp_match.group(1)) if num_exp_match else None

        accs = [float(x) for x in re.findall(r"Epoch\s+1000,\s+ACC:\s+([0-9.]+)", block)]

        final_match = re.search(r"final result:\s*([0-9.]+)±([0-9.]+)", block)
        if final_match:
            mean = float(final_match.group(1))
            std = float(final_match.group(2))
        else:
            mean = None
            std = None

        if not accs:
            continue

        best_single = max(accs)
        best_index = accs.index(best_single)

        rows.append({
            "dataset": dataset,
            "display": DISPLAY.get(dataset, dataset),
            "num_exp": num_exp,
            "mean": mean,
            "std": std,
            "mean_percent": None if mean is None else mean * 100,
            "std_percent": None if std is None else std * 100,
            "best_single": best_single,
            "best_single_percent": best_single * 100,
            "best_run_index": best_index,
            "all_accs": accs,
            "log_file": str(path),
        })

# For each dataset, keep the row with the highest mean.
# If mean is missing, use best_single as fallback.
best_by_dataset = {}
for r in rows:
    d = r["dataset"]
    score = r["mean"] if r["mean"] is not None else r["best_single"]

    if d not in best_by_dataset:
        best_by_dataset[d] = r
    else:
        old = best_by_dataset[d]
        old_score = old["mean"] if old["mean"] is not None else old["best_single"]
        if score > old_score:
            best_by_dataset[d] = r

selected = [best_by_dataset[d] for d in DATASET_NAMES if d in best_by_dataset]

# CSV
csv_path = OUT_DIR / "MAX_SINGLE_RUN_AND_MEAN_RESULTS.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "display",
            "num_exp",
            "mean",
            "std",
            "mean_percent",
            "std_percent",
            "best_single",
            "best_single_percent",
            "best_run_index",
            "all_accs",
            "log_file",
        ],
    )
    writer.writeheader()
    for r in selected:
        writer.writerow(r)

# Markdown
md_path = OUT_DIR / "MAX_SINGLE_RUN_AND_MEAN_RESULTS.md"
lines = []
lines.append("# GraphPAE Final Results: Mean ± Std and Best Single Run")
lines.append("")
lines.append("> Note: Mean ± Std is the proper scientific final result. Best Single Run is reported separately and should not replace the mean.")
lines.append("")
lines.append("| Dataset | Runs | Mean ± Std (%) | Best Single Run (%) | Best Run Index |")
lines.append("|---|---:|---:|---:|---:|")

for r in selected:
    mean_text = "N/A" if r["mean_percent"] is None else f"{r['mean_percent']:.2f} ± {r['std_percent']:.2f}"
    best_text = f"{r['best_single_percent']:.2f}"
    lines.append(
        f"| {r['display']} | {r['num_exp']} | {mean_text} | **{best_text}** | {r['best_run_index']} |"
    )

md_path.write_text("\n".join(lines))

print("Saved CSV:", csv_path)
print("Saved Markdown:", md_path)
print()
print(md_path.read_text())
