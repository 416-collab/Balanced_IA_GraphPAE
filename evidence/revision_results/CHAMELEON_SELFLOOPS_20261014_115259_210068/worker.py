
import subprocess
import sys
from pathlib import Path

python, repo, folder, method = sys.argv[1:]
folder = Path(folder)

for loops in ["keep", "drop"]:
    print(f"START: {method}, self_loops={loops}", flush=True)
    logfile = folder / f"{method}_{loops}.log"
    with logfile.open("x") as output:
        result = subprocess.run(
            [python, "-u", "train_node.py",
             "--dataset", "chameleon", "--num_exp", "5",
             "--sampler", method, "--budget", "0.10",
             "--self-loops", loops],
            cwd=repo, stdout=output, stderr=subprocess.STDOUT
        )
    print(
        f"FINISH: {method}, self_loops={loops}, return_code={result.returncode}",
        flush=True
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

print("WORKER COMPLETE", flush=True)
