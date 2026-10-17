from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

cfg = Path("config") / f"{args.dataset}.yaml"
backup = Path("config") / f"{args.dataset}.yaml.before_ia_budget"

if backup.exists():
    cfg.write_text(backup.read_text())
    print("Restored:", cfg)
else:
    print("No backup found:", backup)
