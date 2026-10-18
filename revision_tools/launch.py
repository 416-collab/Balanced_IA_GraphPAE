#!/usr/bin/env python3
"""Launch one PRL-revision batch on both GPUs (final code + fixed-random patch)."""
from pathlib import Path
from datetime import datetime
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import yaml

ROOT = Path("/home/insuper/GraphPAE_Project/PRL_Revision_20261014_004139_949694")
SMALL = ("cora", "citeseer", "actor", "chameleon", "squirrel")
SAMPLERS = ("random", "spectral", "degree", "ia", "bins_random", "ia_no_bins",
            "ia_raw_degree", "bins_spectral", "bins_degree", "random_fixed")
WEIGHT = {"cora": 1.0, "citeseer": 1.3, "actor": 1.8,
          "chameleon": 2.5, "squirrel": 2.5}
SKIP = {"device", "feat_dim", "pe_sampler", "pe_budget_ratio",
        "pe_keep_self_loops", "dataset", "num_exp", "root", "sampler",
        "budget", "self_loops", "seed", "selection_epoch"}
EXPECT = {"enc_gnn_layer": 3, "embed_dim": 1024, "heads": 4,
          "epochs": 1000, "epochs_eval": 2000, "max_pe_edges": 0}

OLD_CHOICE = "'bins_spectral', 'bins_degree'"
NEW_CHOICE = "'bins_spectral', 'bins_degree', 'random_fixed'"
OLD_SEED = ("selection_seed = int(run_args.seed) * 1000003 + "
            "int(run_args.selection_epoch)")
NEW_SEED = ("selection_seed = int(run_args.seed) * 1000003 + (0 if "
            "str(getattr(run_args, 'pe_sampler', '')) == 'random_fixed' "
            "else int(run_args.selection_epoch))")

WRAPPER = '''


# --- PRL revision: fixed-random control ------------------------------------
# 'random_fixed' calls the ordinary random selector. train_node.py gives it the
# same selection seed at every epoch, so one random subset is drawn per run and
# kept for all pretraining epochs. All other samplers are unchanged.
_prl_select_original = select_budgeted_pe_edges


def select_budgeted_pe_edges(edge_index_pe, u, graph_edge_index=None,
                             args=None, device=None):
    if args is not None and str(getattr(args, "pe_sampler", "")) == "random_fixed":
        import copy
        local_args = copy.copy(args)
        local_args.pe_sampler = "random"
        return _prl_select_original(edge_index_pe, u,
                                    graph_edge_index=graph_edge_index,
                                    args=local_args, device=device)
    return _prl_select_original(edge_index_pe, u,
                                graph_edge_index=graph_edge_index,
                                args=args, device=device)
'''

WORKER = '''import json
import subprocess
import sys
from pathlib import Path

python, batch, jobfile = sys.argv[1:]
batch = Path(batch)
failed = []
for job in json.loads(Path(jobfile).read_text()):
    tag = job["tag"]
    print("START:", tag, flush=True)
    with (batch / (tag + ".log")).open("x") as out:
        result = subprocess.run(
            [python, "-u", "train_node.py", "--dataset", job["dataset"],
             "--num_exp", "5", "--sampler", job["sampler"],
             "--budget", repr(job["budget"])],
            cwd=batch, stdout=out, stderr=subprocess.STDOUT)
    print("FINISH:", tag, "return_code =", result.returncode, flush=True)
    if result.returncode != 0:
        failed.append(tag)
print("FAILED:", ", ".join(failed) if failed else "none", flush=True)
print("WORKER COMPLETE", flush=True)
'''

SELFTEST = r'''
import sys
from types import SimpleNamespace
import torch
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils import load_config
import ia_utils

name = sys.argv[1]
device = torch.device("cuda:0")
data = torch.load(f"dataset/{name}.pt", map_location="cpu", weights_only=False)
cfg = dict(load_config(f"config/{name}.yaml"))
u = data.u[:, :int(cfg.get("max_freqs", 50))].float().to(device)
edge = data.edge_index.long().to(device)
candidate, _ = remove_self_loops(edge)
candidate, _ = add_self_loops(candidate, num_nodes=u.size(0))
cfg.update(device=device, pe_keep_self_loops=True, max_pe_edges=0,
           pe_budget_ratio=0.10)

def pick(mode, seed):
    torch.manual_seed(seed)
    args = SimpleNamespace(**dict(cfg, pe_sampler=mode))
    return ia_utils.select_budgeted_pe_edges(
        candidate, u, graph_edge_index=edge, args=args)

a = pick("random_fixed", 11)
b = pick("random_fixed", 11)
c = pick("random_fixed", 12)
r = pick("random", 11)
assert torch.equal(a, b), "same seed gave different subsets"
assert not torch.equal(a, c), "different seeds gave the same subset"
assert torch.equal(a, r), "random_fixed is not the ordinary random selector"
print("FIXED-RANDOM SELF-TEST: PASS")
'''


def stamp(run):
    return "_".join(run.name.split("_")[-3:])


def find_reference(dataset):
    found = []
    for path in (ROOT / "revision_results").rglob("effective_config.json"):
        run = path.parent
        if not (run / "summary.json").is_file():
            continue
        try:
            cfg = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if cfg.get("dataset") != dataset:
            continue
        if cfg.get("pe_sampler") not in ("random", "ia"):
            continue
        if cfg.get("pe_pair_control") or cfg.get("pe_keep_self_loops") is not True:
            continue
        ratio = float(cfg.get("pe_budget_ratio") or 0.0)
        if min(abs(ratio - 0.10), abs(ratio - 0.25)) > 1e-9:
            continue
        source = run.parents[1] / "config" / f"{dataset}.yaml"
        if source.is_file():
            found.append((stamp(run), run, cfg, source))
    if not found:
        raise SystemExit(f"STOP: no completed final-code run found for {dataset}.")
    found.sort(key=lambda item: item[0])
    return found[-1][1:]


def run_checked(cmd, cwd, log, timeout):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    with log.open("x") as out:
        result = subprocess.run(cmd, cwd=cwd, env=env, stdout=out,
                                stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode, log.read_text(errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--jobs", nargs="+", required=True,
                        help="dataset:sampler:budget")
    opts = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", opts.name):
        raise SystemExit("STOP: --name may contain only letters, digits and _.")

    jobs = []
    for spec in opts.jobs:
        try:
            dataset, sampler, budget = spec.split(":")
            budget = float(budget)
        except ValueError:
            raise SystemExit(f"STOP: bad job '{spec}' (use dataset:sampler:budget).")
        if dataset not in SMALL or sampler not in SAMPLERS or not 0 < budget <= 1:
            raise SystemExit(f"STOP: unsupported job '{spec}'.")
        jobs.append({"dataset": dataset, "sampler": sampler, "budget": budget,
                     "tag": f"{dataset}_{sampler}_{budget:.2f}"})
    tags = [job["tag"] for job in jobs]
    if len(set(tags)) != len(tags):
        raise SystemExit("STOP: duplicate jobs in the list.")

    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True).strip()
    if busy:
        raise SystemExit("STOP: a GPU job is already running. Wait for the "
                         "previous batch (check status), then launch again.")

    runner_text = (ROOT / "train_node.py").read_text()
    if runner_text.count(OLD_CHOICE) != 1 or runner_text.count(OLD_SEED) != 1:
        raise SystemExit("STOP: root train_node.py differs from the expected "
                         "version; nothing launched.")
    if "def select_budgeted_pe_edges(" not in (ROOT / "ia_utils.py").read_text():
        raise SystemExit("STOP: ia_utils.py differs from the expected version.")

    refs = {}
    for dataset in sorted({job["dataset"] for job in jobs}):
        run, cfg, source = find_reference(dataset)
        data = yaml.safe_load(source.read_text())
        bad = sorted(k for k, v in data.items()
                     if k not in SKIP and k in cfg and cfg[k] != v)
        if bad:
            raise SystemExit(f"STOP: {source} no longer matches {run.name}: {bad}")
        wrong = {k: cfg.get(k) for k, v in EXPECT.items() if cfg.get(k) != v}
        if wrong:
            raise SystemExit(f"STOP: {dataset} differs from the paper setup: {wrong}")
        refs[dataset] = (run, source, data)

    batch = ROOT / "revision_results" / f"{opts.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    (batch / "config").mkdir(parents=True)
    (batch / "dataset").mkdir()
    for path in ROOT.glob("*.py"):
        shutil.copy2(path, batch / path.name)

    provenance = {}
    print("REFERENCE CONFIGS (copied unchanged):")
    for dataset, (run, source, data) in refs.items():
        shutil.copy2(source, batch / "config" / f"{dataset}.yaml")
        (batch / "dataset" / f"{dataset}.pt").symlink_to(
            (ROOT / "dataset" / f"{dataset}.pt").resolve(strict=True))
        provenance[dataset] = {
            "reference_run": str(run), "yaml_source": str(source),
            "yaml_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        noise = "adaptive" if data.get("learnable_noise") else data.get("noise_val")
        print(f"  {dataset:<10} ref={run.name} | mask={data.get('mask_ratio')} "
              f"pe_loss={data.get('masked_pe_loss')} alpha_l={data.get('alpha_l')} "
              f"noise={noise} dropout={data.get('gnn_dropout')} "
              f"lr={data.get('init_lr')} wd_eval={data.get('wd_eval')}")

    runner = batch / "train_node.py"
    text = runner.read_text().replace(OLD_CHOICE, NEW_CHOICE).replace(OLD_SEED, NEW_SEED)
    ast.parse(text)
    runner.write_text(text)
    selector = batch / "ia_utils.py"
    text = selector.read_text() + WRAPPER
    ast.parse(text)
    selector.write_text(text)

    (batch / "REFERENCE_CONFIGS.json").write_text(json.dumps(provenance, indent=2))
    (batch / "JOBS.json").write_text(json.dumps(jobs, indent=2))
    (batch / "worker.py").write_text(WORKER)
    (batch / "PATCH_NOTE.txt").write_text(
        "Batch code = root source + 'random_fixed' sampler (same selection seed "
        "every epoch). Other samplers are unchanged.\n")

    if any(job["sampler"] == "random_fixed" for job in jobs):
        name = "cora" if "cora" in refs else min(refs, key=WEIGHT.get)
        code, out = run_checked([sys.executable, "-c", SELFTEST, name],
                                batch, batch / "_selftest.log", 900)
        if code != 0 or "SELF-TEST: PASS" not in out:
            print("\n".join(out.splitlines()[-20:]))
            raise SystemExit("STOP: fixed-random self-test failed; no jobs started.")
        print("FIXED-RANDOM SELF-TEST: PASS")
        smoke = batch / "_smoke"
        (smoke / "config").mkdir(parents=True)
        (smoke / "dataset").mkdir()
        for path in batch.glob("*.py"):
            shutil.copy2(path, smoke / path.name)
        quick = yaml.safe_load((batch / "config" / f"{name}.yaml").read_text())
        quick.update(epochs=2, epochs_eval=5)
        (smoke / "config" / f"{name}.yaml").write_text(
            yaml.safe_dump(quick, sort_keys=False))
        (smoke / "dataset" / f"{name}.pt").symlink_to(
            (batch / "dataset" / f"{name}.pt").resolve(strict=True))
        code, out = run_checked(
            [sys.executable, "-u", "train_node.py", "--dataset", name,
             "--num_exp", "1", "--sampler", "random_fixed", "--budget", "0.1"],
            smoke, smoke / "smoke.log", 1800)
        if code != 0 or "final result" not in out:
            print("\n".join(out.splitlines()[-25:]))
            raise SystemExit("STOP: 2-epoch smoke run failed; no jobs started.")
        print("SMOKE RUN (2 epochs, random_fixed): PASS")

    loads, groups = [0.0, 0.0], [[], []]
    for job in sorted(jobs, key=lambda j: -WEIGHT[j["dataset"]] * (1 + j["budget"])):
        gpu = 0 if loads[0] <= loads[1] else 1
        groups[gpu].append(job)
        loads[gpu] += WEIGHT[job["dataset"]] * (1 + job["budget"])

    for gpu, group in enumerate(groups):
        if not group:
            continue
        jobfile = batch / f"gpu{gpu}_jobs.json"
        jobfile.write_text(json.dumps(group, indent=2))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        with (batch / f"gpu{gpu}_worker.log").open("x") as out:
            proc = subprocess.Popen(
                [sys.executable, str(batch / "worker.py"), sys.executable,
                 str(batch), str(jobfile)],
                cwd=batch, env=env, stdin=subprocess.DEVNULL, stdout=out,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(f"GPU {gpu} | PID {proc.pid} | " + ", ".join(j["tag"] for j in group))

    print("BATCH:", batch)
    print(f"STARTED: {len(jobs)} configs x 5 seeds (background; Xshell can be closed)")
    print(f"STATUS:  python revision_tools/status.py {opts.name}")
    print(f"RESULTS: python revision_tools/collect.py {opts.name}")


if __name__ == "__main__":
    main()
