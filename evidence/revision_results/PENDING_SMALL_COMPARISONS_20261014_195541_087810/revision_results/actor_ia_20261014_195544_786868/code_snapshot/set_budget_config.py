import argparse
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--sampler", required=True, choices=["full", "random", "spectral", "degree", "ia"])
parser.add_argument("--ratio", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=-1)
parser.add_argument("--tag", default="")
args = parser.parse_args()

cfg_path = Path("config") / f"{args.dataset}.yaml"
if not cfg_path.exists():
    raise FileNotFoundError(f"Missing config file: {cfg_path}")

backup_path = Path("config") / f"{args.dataset}.yaml.before_ia_budget"
if not backup_path.exists():
    backup_path.write_text(cfg_path.read_text())

cfg = yaml.safe_load(cfg_path.read_text())

cfg["pe_sampler"] = args.sampler
cfg["pe_budget_ratio"] = float(args.ratio)
cfg["max_pe_edges"] = 0
cfg["ia_beta_spec"] = 0.7
cfg["ia_beta_degree"] = 0.3
cfg["ia_temperature"] = 1.0
cfg["ia_stochastic"] = 0
cfg["ia_max_freqs_for_score"] = int(cfg.get("max_freqs", 50))
cfg["ia_bins"] = 10

if args.epochs > 0:
    cfg["epochs"] = int(args.epochs)

cfg.setdefault("device", 0)
cfg.setdefault("use_amp", False)
cfg.setdefault("amp_dtype", "bf16")
cfg.setdefault("use_checkpoint", False)

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

print("Updated:", cfg_path)
print("dataset:", args.dataset)
print("sampler:", args.sampler)
print("ratio:", args.ratio)
print("epochs:", cfg.get("epochs"))
print("layers:", cfg.get("enc_gnn_layer"))
print("embed_dim:", cfg.get("embed_dim"))
print("heads:", cfg.get("heads"))
print("max_freqs:", cfg.get("max_freqs"))
