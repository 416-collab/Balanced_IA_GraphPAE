import re
import os
from pathlib import Path

DIR = Path(os.path.expanduser("~/GraphPAE_Project/results_server/IA_BUDGET_SCREEN"))
pattern = re.compile(r"final result:\s*([0-9.]+)\s*±\s*([0-9.]+)")

rows = []

for fp in sorted(DIR.glob("*.txt")):
    txt = fp.read_text(errors="ignore")
    m = pattern.findall(txt)
    if not m:
        rows.append((fp.name, None, None))
    else:
        mean, std = m[-1]
        rows.append((fp.name, float(mean) * 100, float(std) * 100))

print("\n=== SELECTOR SCREEN SUMMARY ===")
for name, mean, std in rows:
    if mean is None:
        print(f"{name:60s} ---")
    else:
        print(f"{name:60s} {mean:6.2f} ± {std:5.2f}")

out = DIR / "SELECTOR_SCREEN_SUMMARY.md"
with out.open("w") as f:
    f.write("# Selector Screen Summary\n\n")
    f.write("| Log file | Accuracy (%) |\n")
    f.write("|---|---:|\n")
    for name, mean, std in rows:
        if mean is None:
            f.write(f"| {name} | - |\n")
        else:
            f.write(f"| {name} | {mean:.2f} ± {std:.2f} |\n")

print("\nSaved:", out)
