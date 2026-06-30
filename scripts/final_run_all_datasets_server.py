import os
import re
import csv
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

REPO = Path.cwd()
FINAL_ROOT = Path.home() / "GraphPAE_Project/results_server/FINAL_ALL_DATASETS"
LOG_DIR = FINAL_ROOT / "logs"
CFG_DIR = FINAL_ROOT / "configs"
SUMMARY_DIR = FINAL_ROOT / "summary"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CFG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

BASE_SMALL = {
    "device": 0,
    "feat_dim": 1,
    "epochs": 1000,
    "optim": "adam",
    "init_lr": 0.001,
    "weight_decay": 0.0,
    "use_schedule": True,
    "step_size": 50,
    "gamma": 0.5,
    "lr_eval": 0.01,
    "wd_eval": 0.0001,
    "epochs_eval": 2000,
    "mask_ratio": 0.25,
    "replace_ratio": 0.0,
    "masked_atom_loss": 1.0,
    "masked_pe_loss": 0.01,
    "atom_recon_type": "sce",
    "alpha_l": 2.0,
    "noise_val": 0.01,
    "enc_gnn_layer": 2,
    "embed_dim": 512,
    "heads": 4,
    "gnn_dropout": 0.0,
    "gnn_edge_dropout": 0.0,
    "gnn_activation": "elu",
    "task_head_activation": "gelu",
    "max_freqs": 50,
    "pos_activation": "gelu",
    "learnable_noise": False,
    "use_amp": False,
    "amp_dtype": "bf16",
    "use_checkpoint": False,
}

def cfg(**kw):
    c = dict(BASE_SMALL)
    c.update(kw)
    return c

# Final chosen settings based on previous results.
RUNS = [
    # Original GraphPAE node datasets
    {
        "name": "blog",
        "display": "BlogCatalog",
        "num_exp": 10,
        "tag": "fixed_best",
        "config": cfg(
            lr_eval=0.005,
            mask_ratio=0.25,
            replace_ratio=0.2,
            masked_pe_loss=0.1,
            alpha_l=3.0,
            enc_gnn_layer=2,
            embed_dim=1024,
            heads=4,
            gnn_dropout=0.6,
            learnable_noise=False,
        ),
    },
    {
        "name": "chameleon",
        "display": "Chameleon",
        "num_exp": 10,
        "tag": "adaptive_best",
        "config": cfg(
            feat_dim=2325,
            wd_eval=0.0,
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=3,
            embed_dim=1024,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=True,
        ),
    },
    {
        "name": "squirrel",
        "display": "Squirrel",
        "num_exp": 10,
        "tag": "fixed_best",
        "config": cfg(
            feat_dim=2089,
            wd_eval=0.0,
            mask_ratio=0.5,
            masked_pe_loss=0.001,
            alpha_l=1.0,
            enc_gnn_layer=2,
            embed_dim=1024,
            heads=4,
            gnn_dropout=0.6,
            learnable_noise=False,
        ),
    },
    {
        "name": "actor",
        "display": "Actor",
        "num_exp": 10,
        "tag": "adaptive_best",
        "config": cfg(
            feat_dim=932,
            init_lr=0.0005,
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=2,
            embed_dim=1024,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=True,
        ),
    },

    # Extra datasets
    {
        "name": "texas",
        "display": "Texas",
        "num_exp": 10,
        "tag": "fixed_best",
        "config": cfg(
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=2,
            embed_dim=512,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=False,
        ),
    },
    {
        "name": "cornell",
        "display": "Cornell",
        "num_exp": 10,
        "tag": "fixed_best",
        "config": cfg(
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=2,
            embed_dim=512,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=False,
        ),
    },
    {
        "name": "wisconsin",
        "display": "Wisconsin",
        "num_exp": 10,
        "tag": "fixed_best",
        "config": cfg(
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=2,
            embed_dim=512,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=False,
        ),
    },
    {
        "name": "cora",
        "display": "Cora",
        "num_exp": 10,
        "tag": "fixed_best",
        "config": cfg(
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=2,
            embed_dim=512,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=False,
        ),
    },
    {
        "name": "citeseer",
        "display": "CiteSeer",
        "num_exp": 10,
        "tag": "adaptive_best",
        "config": cfg(
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=2,
            embed_dim=512,
            heads=4,
            gnn_dropout=0.0,
            learnable_noise=True,
        ),
    },

    # Large server datasets
    {
        "name": "penn94",
        "display": "Penn94",
        "num_exp": 10,
        "tag": "memory_safe_final",
        "config": cfg(
            mask_ratio=0.25,
            masked_pe_loss=0.01,
            alpha_l=2.0,
            enc_gnn_layer=1,
            embed_dim=64,
            heads=1,
            gnn_dropout=0.3,
            gnn_edge_dropout=0.5,
            max_freqs=50,
            max_pe_edges=200000,
            pe_loss_mode="random",
            pe_weight_mode="none",
            learnable_noise=False,
            use_amp=True,
            amp_dtype="bf16",
            use_checkpoint=True,
        ),
    },
    {
        "name": "arxiv-year-official",
        "display": "arXiv-year Official",
        # Official file has 5 splits. Keep 5 for clean official result.
        "num_exp": 5,
        "tag": "official_final_3L256H2",
        "config": cfg(
            mask_ratio=0.5,
            masked_pe_loss=0.05,
            alpha_l=2.0,
            enc_gnn_layer=3,
            embed_dim=256,
            heads=2,
            gnn_dropout=0.3,
            gnn_edge_dropout=0.5,
            max_freqs=100,
            max_pe_edges=500000,
            pe_loss_mode="random",
            pe_weight_mode="none",
            learnable_noise=False,
            use_amp=True,
            amp_dtype="bf16",
            use_checkpoint=True,
        ),
    },
]

def write_config(dataset, config):
    path = REPO / "config" / f"{dataset}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    shutil.copy2(path, CFG_DIR / f"{dataset}_FINAL.yaml")

def parse_final_result(text):
    m = re.search(r"final result:\s*([0-9.]+)±([0-9.]+)", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))

def parse_acc_list(text):
    matches = re.findall(r"Epoch\s+1000,\s+ACC:\s+([0-9.]+)", text)
    return [float(x) for x in matches]

def main():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    rows = []
    started = datetime.now().isoformat(timespec="seconds")
    print("FINAL RUN START:", started)
    print("Output folder:", FINAL_ROOT)

    # dataset existence check
    missing = []
    for r in RUNS:
        data_path = REPO / "dataset" / f"{r['name']}.pt"
        if not data_path.exists():
            missing.append(r["name"])
            print("MISSING:", data_path)
        else:
            print("DATA OK:", data_path)
    if missing:
        raise SystemExit(f"Missing dataset files: {missing}")

    for i, r in enumerate(RUNS, 1):
        dataset = r["name"]
        display = r["display"]
        tag = r["tag"]
        num_exp = r["num_exp"]

        print("\n" + "=" * 100)
        print(f"[{i}/{len(RUNS)}] Running {display} ({dataset}) | {tag} | num_exp={num_exp}")
        print("=" * 100)

        write_config(dataset, r["config"])

        log_path = LOG_DIR / f"{dataset}_{tag}_numexp{num_exp}.txt"
        cmd = ["python", "-u", "train_node.py", "--dataset", dataset, "--num_exp", str(num_exp)]

        with open(log_path, "w") as f:
            process = subprocess.Popen(
                cmd,
                cwd=str(REPO),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                print(line, end="")
                f.write(line)

            process.wait()
            return_code = process.returncode

        text = log_path.read_text(errors="ignore")
        mean, std = parse_final_result(text)
        accs = parse_acc_list(text)

        rows.append({
            "dataset": dataset,
            "display": display,
            "tag": tag,
            "num_exp": num_exp,
            "return_code": return_code,
            "mean": mean,
            "std": std,
            "mean_percent": None if mean is None else round(mean * 100, 4),
            "std_percent": None if std is None else round(std * 100, 4),
            "acc_list": accs,
            "log_file": str(log_path),
        })

        # update summary after every dataset
        write_summary(rows)

        if return_code != 0:
            print(f"\nERROR: {dataset} failed with return code {return_code}.")
            print("Fix this dataset, then rerun the script or continue manually.")
            raise SystemExit(return_code)

    print("\nALL FINAL RUNS COMPLETED.")
    write_summary(rows)

def write_summary(rows):
    csv_path = SUMMARY_DIR / "FINAL_ALL_RESULTS.csv"
    md_path = SUMMARY_DIR / "FINAL_ALL_RESULTS.md"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "display", "tag", "num_exp", "return_code",
                "mean", "std", "mean_percent", "std_percent", "acc_list", "log_file"
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = []
    lines.append("# Final GraphPAE Results\n")
    lines.append("| Dataset | Setting | Runs | Result (%) | Status |")
    lines.append("|---|---|---:|---:|---|")
    for r in rows:
        if r["mean_percent"] is None:
            res = "FAILED / not parsed"
        else:
            res = f"{r['mean_percent']:.2f} ± {r['std_percent']:.2f}"
        status = "OK" if r["return_code"] == 0 else f"ERROR {r['return_code']}"
        lines.append(f"| {r['display']} | {r['tag']} | {r['num_exp']} | {res} | {status} |")
    md_path.write_text("\n".join(lines))

if __name__ == "__main__":
    main()
