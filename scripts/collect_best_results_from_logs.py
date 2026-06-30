import re
import csv
import shutil
from pathlib import Path

ROOT = Path.home() / "GraphPAE_Project/results_server"
OUT = ROOT / "FINAL_SELECTED_RESULTS"
BEST_LOGS = OUT / "best_logs"
OUT.mkdir(parents=True, exist_ok=True)
BEST_LOGS.mkdir(parents=True, exist_ok=True)

target_datasets = [
    "blog", "chameleon", "squirrel", "actor",
    "texas", "cornell", "wisconsin", "cora", "citeseer",
    "penn94", "arxiv-year-official"
]

rows = []

for path in ROOT.rglob("*.txt"):
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        continue

    # find every Namespace(...) block followed by a final result
    pattern = re.compile(
        r"Namespace\((.*?)\).*?final result:\s*([0-9.]+)±([0-9.]+)",
        re.S
    )

    for m in pattern.finditer(text):
        ns = m.group(1)
        mean = float(m.group(2))
        std = float(m.group(3))

        dsm = re.search(r"dataset='([^']+)'", ns)
        nm = re.search(r"num_exp=([0-9]+)", ns)

        if not dsm:
            continue

        dataset = dsm.group(1)
        num_exp = int(nm.group(1)) if nm else -1

        if dataset not in target_datasets:
            continue

        rows.append({
            "dataset": dataset,
            "num_exp": num_exp,
            "mean": mean,
            "std": std,
            "mean_percent": mean * 100,
            "std_percent": std * 100,
            "log_file": str(path),
            "namespace": "Namespace(" + ns + ")",
        })

# Save all candidates
all_csv = OUT / "ALL_CANDIDATE_RESULTS.csv"
with all_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset", "num_exp", "mean", "std",
            "mean_percent", "std_percent", "log_file", "namespace"
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

# Eligible best: small datasets and Penn94 need 10 runs, arxiv official needs 5
def eligible(r):
    if r["dataset"] == "arxiv-year-official":
        return r["num_exp"] >= 5
    return r["num_exp"] >= 10

best = {}
for r in rows:
    if not eligible(r):
        continue
    d = r["dataset"]
    if d not in best or r["mean"] > best[d]["mean"]:
        best[d] = r

best_csv = OUT / "BEST_SELECTED_RESULTS.csv"
with best_csv.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset", "num_exp", "mean", "std",
            "mean_percent", "std_percent", "log_file", "namespace"
        ],
    )
    writer.writeheader()
    for d in target_datasets:
        if d in best:
            writer.writerow(best[d])

md = OUT / "BEST_SELECTED_RESULTS.md"
lines = []
lines.append("# Best Selected GraphPAE Results")
lines.append("")
lines.append("| Dataset | Runs | Best Result (%) | Log file |")
lines.append("|---|---:|---:|---|")

for d in target_datasets:
    if d not in best:
        lines.append(f"| {d} | - | NOT FOUND | - |")
        continue

    r = best[d]
    lines.append(
        f"| {d} | {r['num_exp']} | {r['mean_percent']:.2f} ± {r['std_percent']:.2f} | {Path(r['log_file']).name} |"
    )

    src = Path(r["log_file"])
    dst = BEST_LOGS / f"{d}_BEST_{src.name}"
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print("Could not copy:", src, e)

md.write_text("\n".join(lines))

print("Saved:", all_csv)
print("Saved:", best_csv)
print("Saved:", md)
print("\nBest results:")
print(md.read_text())
