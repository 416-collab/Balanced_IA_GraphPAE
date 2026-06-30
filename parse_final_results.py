import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "FINAL_RESULTS_SUMMARY.md"

pattern = re.compile(r"final result:\s*([0-9.]+)\s*±\s*([0-9.]+)")

def parse_dir(folder, keyword):
    rows = {}
    for fp in sorted(folder.glob("*.txt")):
        txt = fp.read_text(errors="ignore")
        matches = pattern.findall(txt)
        if not matches:
            continue
        mean, std = matches[-1]
        name = fp.name

        dataset = name.split("_")[0]
        ratio = None
        if "ratio0.25" in name:
            ratio = "0.25"
        elif "ratio0.10" in name:
            ratio = "0.10"

        if dataset and ratio:
            rows[(dataset, ratio)] = (float(mean) * 100, float(std) * 100, fp.name)
    return rows

random_rows = parse_dir(RESULTS / "random_budget_5seed", "random")
ia_rows = parse_dir(RESULTS / "balanced_ia_budget_5seed", "ia")

datasets = ["actor", "chameleon", "citeseer", "cora", "squirrel"]
ratios = ["0.10", "0.25"]

lines = []
lines.append("# Final Budgeted GraphPAE Results\n")
lines.append("| Dataset | Budget | Random (%) | Balanced IA (%) | Gain |\n")
lines.append("|---|---:|---:|---:|---:|\n")

print("\n=== FINAL RANDOM VS BALANCED IA RESULTS ===")
for d in datasets:
    for r in ratios:
        rv = random_rows.get((d, r))
        iv = ia_rows.get((d, r))

        if rv is None or iv is None:
            lines.append(f"| {d} | {r} | - | - | - |\n")
            print(f"{d:10s} ratio={r} missing result")
            continue

        rmean, rstd, _ = rv
        imean, istd, _ = iv
        gain = imean - rmean

        lines.append(
            f"| {d} | {r} | {rmean:.2f} ± {rstd:.2f} | {imean:.2f} ± {istd:.2f} | {gain:+.2f} |\n"
        )
        print(f"{d:10s} ratio={r} random={rmean:.2f}±{rstd:.2f} IA={imean:.2f}±{istd:.2f} gain={gain:+.2f}")

RESULTS.mkdir(exist_ok=True)
OUT.write_text("".join(lines))
print("\nSaved:", OUT)
