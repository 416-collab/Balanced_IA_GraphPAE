
import subprocess
import sys
from pathlib import Path

python, repo, folder, method = sys.argv[1:]
folder = Path(folder)

for ratio in [0.05, 0.25, 0.50]:
    print(f"START: {method}, budget={ratio}", flush=True)
    logfile = folder / f"{method}_budget{ratio:.2f}.log"
    with logfile.open("x") as output:
        result = subprocess.run(
            [python, "-u", "train_node.py",
             "--dataset", "chameleon",
             "--num_exp", "5",
             "--sampler", method,
             "--budget", str(ratio)],
            cwd=repo,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    print(
        f"FINISH: {method}, budget={ratio}, return_code={result.returncode}",
        flush=True
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)

print("WORKER COMPLETE", flush=True)
