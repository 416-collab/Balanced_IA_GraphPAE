import re
import os
from pathlib import Path

RESULTS_DIR = Path(os.path.expanduser("~/GraphPAE_Project/results_server/IA_BUDGET_RUN"))
OUT = RESULTS_DIR / "IA_BUDGET_SUMMARY.md"

pattern = re.compile(r"final result:\s*([0-9.]+)\s*±\s*([0-9.]+)")

rows = []
if RESULTS_DIR.exists():
    files = sorted(RESULTS_DIR.glob("*.txt"))
else:
    files = []

for fp in files:
    txt = fp.read_text(errors="ignore")
    matches = pattern.findall(txt)
    if matches:
        mean, std = matches[-1]
        rows.append((fp.name, float(mean) * 100, float(std) * 100, "OK"))
    else:
        rows.append((fp.name, None, None, "NO FINAL RESULT"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with OUT.open("w") as f:
    f.write("# IA-GraphPAE Budget Experiment Summary\n\n")
    f.write("| Log file | Accuracy (%) | Status |\n")
    f.write("|---|---:|---|\n")
    for name, mean, std, status in rows:
        if mean is None:
            f.write(f"| {name} | - | {status} |\n")
        else:
            f.write(f"| {name} | {mean:.2f} ± {std:.2f} | {status} |\n")

print("\n=== IA BUDGET SUMMARY ===")
for name, mean, std, status in rows:
    if mean is None:
        print(f"{name:60s} --- [{status}]")
    else:
        print(f"{name:60s} {mean:6.2f} ± {std:5.2f} [{status}]")
print("\nSaved:", OUT)
