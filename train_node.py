
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import json
import shutil
import hashlib
import time
import gc
import numpy as np
import torch
from ia_utils import select_budgeted_pe_edges

from utils import set_random_seed, split, load_config, create_optimizer
from evaluation import node_evaluation
from encoder import GraphEncoder
from autoencoder import GraphAutoEncoder

from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
)


def train_mae_epoch(graph_auto_encoder, x, edge_index, edge_index_pe, u, PE, optimizer):
    graph_auto_encoder.train()

    run_args = getattr(graph_auto_encoder, 'args', None)
    if run_args is None:
        run_args = globals().get('args', None)

    # IA/Budgeted GraphPAE uses one active PE-edge set.
    # The same active edge set must be used for message passing,
    # positional reconstruction, and PE edge attributes.
    rng_devices = [x.device.index] if x.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=rng_devices):
        selection_seed = int(run_args.seed) * 1000003 + int(run_args.selection_epoch)
        torch.manual_seed(selection_seed)
        active_edge_index = select_budgeted_pe_edges(
            edge_index_pe=edge_index_pe,
            u=u,
            graph_edge_index=edge_index,
            args=run_args,
            device=x.device,
        )

    # Critical fix: recompute PE values for the selected active edges.
    # Without this, edge_index length and PE length mismatch inside conv.py.
    active_PE = torch.linalg.norm(
        u[active_edge_index[0]] - u[active_edge_index[1]],
        dim=-1,
    )

    if active_PE.size(0) != active_edge_index.size(1):
        raise RuntimeError(
            f'active_PE length {active_PE.size(0)} != active_edge_index length {active_edge_index.size(1)}'
        )

    optimizer.zero_grad(set_to_none=True)

    use_amp = bool(getattr(run_args, 'use_amp', False)) if run_args is not None else False
    amp_dtype_name = getattr(run_args, 'amp_dtype', 'bf16') if run_args is not None else 'bf16'
    amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16

    with torch.autocast(
        device_type='cuda',
        dtype=amp_dtype,
        enabled=(use_amp and x.device.type == 'cuda'),
    ):
        loss = graph_auto_encoder(
            x,
            active_edge_index,
            u,
            active_PE,
            active_edge_index,
        )

    loss.backward()
    optimizer.step()
    return loss.detach()

def select_mask(mask, seed):
    """
    Supports both:
    1D masks: Cora/CiteSeer/Planetoid style
    2D masks: Texas/Cornell/Wisconsin/WebKB style
    """
    if mask.dim() == 1:
        return mask
    elif mask.dim() == 2:
        return mask[:, seed % mask.size(1)]
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.dim()}")


parser = ArgumentParser()
parser.add_argument('--num_exp', type=int, default=5)
parser.add_argument('--root', type=str, default="./dataset")
parser.add_argument('--dataset', type=str, default="blog")
parser.add_argument(
    '--sampler', required=True,
    choices=[
        'random', 'spectral', 'degree', 'ia',
        'bins_random', 'ia_no_bins', 'ia_raw_degree',
        'bins_spectral', 'bins_degree'
    ]
)
parser.add_argument('--budget', type=float, default=None)
parser.add_argument(
    '--self-loops', choices=['keep', 'drop'], default='keep'
)
args = parser.parse_args()
cli_keep_self_loops = args.self_loops == 'keep'
cli_budget = args.budget
if cli_budget is not None and not 0 < cli_budget <= 1:
    parser.error('--budget must be greater than 0 and at most 1')
cli_sampler = args.sampler

config = load_config(f"./config/{args.dataset}.yaml")
for key, value in config.items():
    setattr(args, key, value)

args.pe_sampler = cli_sampler
args.pe_keep_self_loops = cli_keep_self_loops
if cli_budget is not None:
    args.pe_budget_ratio = cli_budget

args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

print(args)

# Safe loading for PyTorch 2.6+
data = torch.load('./dataset/{}.pt'.format(args.dataset), weights_only=False)

x = data.x.float().to(args.device)
args.feat_dim = x.shape[1]
print("Auto feat_dim:", args.feat_dim)

run_dir = Path("revision_results") / (
    f"{args.dataset}_{args.pe_sampler}_"
    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
run_dir.mkdir(parents=True, exist_ok=False)
effective_config = {
    k: str(v) if isinstance(v, torch.device) else v
    for k, v in vars(args).items()
}
(run_dir / "effective_config.json").write_text(
    json.dumps(effective_config, indent=2)
)
code_dir = run_dir / "code_snapshot"
code_dir.mkdir()
for source in Path(".").glob("*.py"):
    shutil.copy2(source, code_dir / source.name)
shutil.copy2(
    Path("config") / f"{args.dataset}.yaml",
    run_dir / "dataset_config.yaml"
)
manifest = {
    source.name: hashlib.sha256(source.read_bytes()).hexdigest()
    for source in code_dir.glob("*.py")
}
(run_dir / "code_hashes.json").write_text(json.dumps(manifest, indent=2))
(run_dir / "environment.json").write_text(json.dumps({
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(args.device)
    if args.device.type == "cuda" else "CPU"
}, indent=2))
print("RUN DIRECTORY:", run_dir.resolve(), flush=True)
seed_results = []

edge = data.edge_index.long().to(args.device)
e = data.e[:args.max_freqs].float().to(args.device)
u = data.u[:, :args.max_freqs].float().to(args.device)

y = data.y.view(-1).long().to(args.device)
print(y.min().item(), y.max().item())
nclass = y.max().item() + 1

edge_index_pe, _ = remove_self_loops(edge, None)
edge_index_pe, _ = add_self_loops(edge_index_pe, fill_value='mean', num_nodes=u.shape[0])
PE = torch.linalg.norm(u[edge_index_pe[0]] - u[edge_index_pe[1]], dim=-1)

final_results = []

for ep_num in range(args.num_exp):
    args.seed = ep_num
    set_random_seed(ep_num)

    if 'train_mask' in data.keys():
        train_mask = select_mask(data.train_mask, args.seed)
        val_mask = select_mask(data.val_mask, args.seed)
        test_mask = select_mask(data.test_mask, args.seed)

        train_idx = torch.where(train_mask)[0]
        val_idx = torch.where(val_mask)[0]
        test_idx = torch.where(test_mask)[0]
    else:
        train_idx, val_idx, test_idx = split(y)

    encoder = GraphEncoder(out_dim=args.embed_dim, args=args).to(args.device)
    model = GraphAutoEncoder(encoder=encoder, num_atom_type=args.feat_dim, args=args).to(args.device)

    parameters = model.parameters()

    if args.optim == "sgd":
        pass
    else:
        args.momentum = None

    optimizer = create_optimizer(
        opt=args.optim,
        parameters=parameters,
        lr=args.init_lr,
        weight_decay=float(args.weight_decay),
        momentum=args.momentum
    )

    if args.use_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    else:
        scheduler = None

    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    pretrain_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        args.selection_epoch = epoch
        epoch_loss = train_mae_epoch(
            graph_auto_encoder=model,
            x=x,
            edge_index=edge,
            u=u,
            PE=PE,
            edge_index_pe=edge_index_pe,
            optimizer=optimizer
        )

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"seed={ep_num} epoch={epoch} loss={float(epoch_loss):.6f}",
                flush=True
            )
        if scheduler:
            scheduler.step()

    if args.device.type == "cuda":
        torch.cuda.synchronize(args.device)
    pretrain_seconds = time.perf_counter() - pretrain_started
    pretrain_peak_gib = (
        torch.cuda.max_memory_allocated(args.device) / 2**30
        if args.device.type == "cuda" else None
    )
    print(
        "PRETRAIN METRICS:",
        "seconds=", pretrain_seconds,
        "peak_allocated_GiB=", pretrain_peak_gib,
        flush=True
    )
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        inference_edges = edge_index_pe
        if not args.pe_keep_self_loops:
            inference_edges, _ = remove_self_loops(inference_edges)
        inference_PE = torch.linalg.norm(
            u[inference_edges[0]] - u[inference_edges[1]], dim=-1
        )
        embed = model.embed(x, inference_edges, inference_PE)

    set_random_seed(10000 + ep_num)
    acc, pred = node_evaluation(
        emb=embed,
        y=y,
        train_idx=train_idx,
        valid_idx=val_idx,
        test_idx=test_idx,
        epochs=args.epochs_eval,
        lr=args.lr_eval,
        weight_decay=args.wd_eval
    )

    if hasattr(model, "get_loss_weights"):
        w = model.get_loss_weights()
        print(f"Epoch {epoch}, ACC: {acc.item()}, atom_w: {w['atom_weight']:.6f}, pe_w: {w['pe_weight']:.6f}, pe_to_atom: {w['pe_to_atom']:.6f}")
    elif hasattr(model, "get_adaptive_noise_stats"):
        s = model.get_adaptive_noise_stats()
        print(f"Epoch {epoch}, ACC: {acc.item()}, noise_mean: {s['mean']:.6f}, noise_min: {s['min']:.6f}, noise_max: {s['max']:.6f}")
    elif hasattr(model, "get_adaptive_noise"):
        print(f"Epoch {epoch}, ACC: {acc.item()}, learned_noise: {model.get_adaptive_noise():.6f}")
    else:
        print(f"Epoch {epoch}, ACC: {acc.item()}")

    final_results.append(acc.item())
    seed_results.append({
        "seed": ep_num,
        "split_column": ep_num % data.train_mask.size(1)
        if data.train_mask.dim() == 2 else None,
        "accuracy": float(acc),
        "pretrain_seconds": pretrain_seconds,
        "pretrain_peak_allocated_gib": pretrain_peak_gib,
        "train_count": int(train_idx.numel()),
        "validation_count": int(val_idx.numel()),
        "test_count": int(test_idx.numel())
    })
    (run_dir / "seed_results.json").write_text(
        json.dumps(seed_results, indent=2)
    )
    torch.save({
        "seed": ep_num,
        "test_indices": test_idx.cpu(),
        "test_predictions": pred.cpu(),
        "test_labels": y[test_idx].cpu()
    }, run_dir / f"predictions_seed_{ep_num}.pt")
    del embed, model, encoder, pred, epoch_loss
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()

mean_final_result = np.mean(final_results)
std_final_result = np.std(final_results)

print(f"{final_results}")
print(f"final result: {mean_final_result:.5f}±{std_final_result:.5}")

(run_dir / "summary.json").write_text(json.dumps({
    "dataset": args.dataset,
    "sampler": args.pe_sampler,
    "runs": len(final_results),
    "accuracies": final_results,
    "mean": float(mean_final_result),
    "std_population": float(std_final_result)
}, indent=2))
