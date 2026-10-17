
from pathlib import Path
import subprocess
import sys
import json

folder = Path(__file__).resolve().parent
method = sys.argv[1]
results = []
for dataset in ["cora", "chameleon"]:
    for budget in [0.10, 0.25]:
        tag = f"{dataset}_{method}_{budget:.2f}"
        print("START:", tag, flush=True)
        with (folder / f"{tag}.log").open("w") as log:
            p = subprocess.run(
                [
                    sys.executable, "train_node.py",
                    "--dataset", dataset, "--sampler", method,
                    "--budget", str(budget),
                    "--self-loops", "keep", "--num_exp", "5"
                ],
                cwd=folder, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT
            )
        results.append({"run": tag, "return_code": p.returncode})
        (folder / f"{method}_status.json").write_text(
            json.dumps(results, indent=2)
        )
        print("FINISH:", tag, "return_code=", p.returncode, flush=True)
        if p.returncode:
            raise SystemExit(p.returncode)
print("WORKER COMPLETE", flush=True)
