import re, os

RESULTS_DIR = os.path.expanduser("~/GraphPAE_Project/results_server/FINAL_PAPER_RUN")
OUT_FILE = os.path.join(RESULTS_DIR, "FINAL_SUMMARY_TABLE.md")
PATTERN = re.compile(r"final result:\s*([\d.]+)\s*±\s*([\d.]+)")

DATASET_ORDER = ["blog", "chameleon", "squirrel", "actor", "texas", "cornell", "wisconsin", "cora", "citeseer"]
DISPLAY_NAME = {
    "blog": "BlogCatalog", "chameleon": "Chameleon", "squirrel": "Squirrel", "actor": "Actor",
    "texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin", "cora": "Cora", "citeseer": "CiteSeer",
}

def parse_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    matches = PATTERN.findall(content)
    if not matches:
        return None
    mean, std = matches[-1]
    return float(mean) * 100, float(std) * 100

rows = []
for name in DATASET_ORDER:
    filepath = os.path.join(RESULTS_DIR, f"{name}_final_10seed.txt")
    if not os.path.exists(filepath):
        rows.append((name, None, None, "FILE NOT FOUND"))
        continue
    result = parse_file(filepath)
    if result is None:
        rows.append((name, None, None, "NO RESULT LINE FOUND"))
    else:
        mean, std = result
        rows.append((name, mean, std, "OK"))

with open(OUT_FILE, "w") as f:
    f.write("# Final Results - Pre-Paper Run\n\n")
    f.write("| Dataset | Final Result (%) | Status |\n")
    f.write("|---|---:|---|\n")
    for name, mean, std, status in rows:
        display = DISPLAY_NAME.get(name, name)
        if mean is not None:
            f.write(f"| {display} | {mean:.2f} +/- {std:.2f} | {status} |\n")
        else:
            f.write(f"| {display} | - | {status} |\n")

print("\n=== FINAL SUMMARY TABLE ===")
for name, mean, std, status in rows:
    display = DISPLAY_NAME.get(name, name)
    if mean is not None:
        print(f"{display:12s} {mean:6.2f} +/- {std:5.2f}   [{status}]")
    else:
        print(f"{display:12s}  ---   ---   [{status}]")

print(f"\nSaved to: {OUT_FILE}")
