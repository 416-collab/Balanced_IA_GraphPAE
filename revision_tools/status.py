#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path("/home/insuper/GraphPAE_Project/PRL_Revision_20261014_004139_949694")
if len(sys.argv) != 2:
    raise SystemExit("usage: python revision_tools/status.py NAME")
batches = sorted(p for p in (ROOT / "revision_results").glob(sys.argv[1] + "_*")
                 if (p / "JOBS.json").is_file())
if not batches:
    raise SystemExit("No batch found for " + sys.argv[1])
batch = batches[-1]
print("BATCH:", batch)
for log in sorted(batch.glob("gpu*_worker.log")):
    lines = log.read_text(errors="replace").splitlines()
    print(f"{log.name}: {lines[-1] if lines else 'empty'}")
print()
jobs = json.loads((batch / "JOBS.json").read_text())
done = 0
for job in jobs:
    log = batch / (job["tag"] + ".log")
    if not log.exists():
        state = "waiting"
    else:
        text = log.read_text(errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        final = [line for line in lines if line.startswith("final result:")]
        if final:
            state = "DONE    " + final[-1]
            done += 1
        elif "Traceback (most recent call last)" in text:
            state = "FAILED  see " + log.name
        else:
            state = "running " + (lines[-1][:80] if lines else "")
    print(f"{job['tag']:<34} {state}")
print(f"\nCOMPLETED: {done}/{len(jobs)}")
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
