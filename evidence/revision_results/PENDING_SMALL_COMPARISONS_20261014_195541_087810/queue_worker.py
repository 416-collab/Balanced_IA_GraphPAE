
from pathlib import Path
import subprocess
import sys
import json

folder = Path(__file__).resolve().parent
method = sys.argv[1]
status = []

for dataset in ["actor", "squirrel", "citeseer"]:
    for budget in [0.10, 0.25]:
        tag = f"{dataset}_{method}_budget{budget:.2f}"
        print("START:", tag, flush=True)
        with (folder / f"{tag}.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable, "train_node.py",
                    "--dataset", dataset,
                    "--sampler", method,
                    "--budget", str(budget),
                    "--self-loops", "keep",
                    "--num_exp", "5",
                ],
                cwd=folder,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status.append({"experiment": tag, "return_code": result.returncode})
        (folder / f"{method}_status.json").write_text(
            json.dumps(status, indent=2)
        )
        print("FINISH:", tag, "return_code=", result.returncode, flush=True)

print("WORKER COMPLETE", flush=True)
