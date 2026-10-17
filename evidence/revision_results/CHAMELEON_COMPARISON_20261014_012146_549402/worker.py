
import subprocess
import sys
from pathlib import Path

python, repo, folder, *methods = sys.argv[1:]
folder = Path(folder)

for method in methods:
    print(f"START: {method}", flush=True)
    with (folder / f"{method}.log").open("x") as output:
        result = subprocess.run(
            [python, "-u", "train_node.py",
             "--dataset", "chameleon", "--num_exp", "5",
             "--sampler", method],
            cwd=repo, stdout=output, stderr=subprocess.STDOUT
        )
    print(f"FINISH: {method}, return_code={result.returncode}", flush=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

print("WORKER COMPLETE", flush=True)
