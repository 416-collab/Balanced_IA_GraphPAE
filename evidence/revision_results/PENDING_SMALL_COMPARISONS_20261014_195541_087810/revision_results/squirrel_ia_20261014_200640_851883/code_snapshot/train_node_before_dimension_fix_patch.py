
from argparse import ArgumentParser
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

    edge_index_pe_epoch = select_budgeted_pe_edges(
        edge_index_pe=edge_index_pe,
        u=u,
        graph_edge_index=edge_index,
        args=run_args,
        device=x.device,
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
        loss = graph_auto_encoder(x, edge_index, u, PE, edge_index_pe_epoch)

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
args = parser.parse_args()

config = load_config(f"./config/{args.dataset}.yaml")
for key, value in config.items():
    setattr(args, key, value)

args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

print(args)

# Safe loading for PyTorch 2.6+
data = torch.load('./dataset/{}.pt'.format(args.dataset), weights_only=False)

x = data.x.float().to(args.device)
args.feat_dim = x.shape[1]
print("Auto feat_dim:", args.feat_dim)

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
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / args.epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    for epoch in range(1, args.epochs + 1):
        train_mae_epoch(
            graph_auto_encoder=model,
            x=x,
            edge_index=edge,
            u=u,
            PE=PE,
            edge_index_pe=edge_index_pe,
            optimizer=optimizer
        )

        if scheduler:
            scheduler.step()

    model.eval()
    with torch.no_grad():
        embed = model.embed(x, edge, PE)

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

mean_final_result = np.mean(final_results)
std_final_result = np.std(final_results)

print(f"{final_results}")
print(f"final result: {mean_final_result:.5f}±{std_final_result:.5}")
