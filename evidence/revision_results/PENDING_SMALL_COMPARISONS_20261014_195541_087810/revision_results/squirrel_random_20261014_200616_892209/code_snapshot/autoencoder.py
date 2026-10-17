from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import remove_self_loops, to_undirected
from utils import get_activation, noise_fn


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, 128)
        self.layer_norm = nn.LayerNorm(128)
        self.out_proj = nn.Linear(128, 1)
        self.activation_fn = get_activation(activation_fn)

    def forward(self, dist, edge_index_pe):
        edge_index, dist = remove_self_loops(edge_index=edge_index_pe, edge_attr=dist)

        dist = self.dense(dist)
        dist = self.activation_fn(dist)
        dist = self.layer_norm(dist)
        dist = self.out_proj(dist)  # [e_sum, head] -> [n_sum, 1]
        
        edge_index, dist = to_undirected(edge_index=edge_index, edge_attr=dist, reduce="mean")        
        return dist.squeeze(), edge_index


class MaskLMHead(nn.Module):
    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, mask_tokens=None):
        if mask_tokens is not None:
            features = features[mask_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias

        return x
        

class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, num_atom_type=0, args=None):
        super(GraphAutoEncoder, self).__init__()
        self.args = args
        self.encoder = encoder

        self.mask_ratio = args.mask_ratio
        self.replace_ratio = args.replace_ratio
        self.noise_val = args.noise_val
        self.learnable_noise = bool(getattr(args, "learnable_noise", False))
        if self.learnable_noise:
            init_noise = max(float(args.noise_val), 1e-8)
            self.noise_logit = nn.Parameter(torch.log(torch.tensor(init_noise, dtype=torch.float32)))
        else:
            self.register_parameter("noise_logit", None)

        self.masked_atom_loss = float(args.masked_atom_loss)
        self.masked_pe_loss = float(args.masked_pe_loss)
        self.atom_recon_type = args.atom_recon_type
        self.num_atom_type = num_atom_type
        self.alpha_l = args.alpha_l

        self.enc_mask_token = nn.Parameter(torch.zeros(1, args.feat_dim))
        self.node_pred = MaskLMHead(args.embed_dim, output_dim=self.num_atom_type, activation_fn=args.task_head_activation)
        self.pe_reconstruct_heads = DistanceHead(heads=args.heads, activation_fn=args.task_head_activation)


    def _current_noise_val(self):
        if self.learnable_noise and self.noise_logit is not None:
            return torch.exp(self.noise_logit).clamp(min=1e-8, max=1.0)
        return float(self.noise_val)

    def forward(self, x, edge_index, u, PE, edge_index_pe=None):
        x_masked, u_masked, mask_tokens = self.encoding_mask_noise(
            x=x,
            u=u,
            mask_ratio=self.mask_ratio,
            replace_ratio=self.replace_ratio,
        )

        # Full PE is kept for encoder, because GAT internally uses full graph edges.
        PE_noise = torch.linalg.norm(
            u_masked[edge_index_pe[0]] - u_masked[edge_index_pe[1]],
            dim=-1,
        )

        enc_rep, pe = self.encoder(
            x,
            x_masked,
            edge_index,
            PE=PE,
            PE_noise=PE_noise,
        )

        enc_rep = self.node_pred(enc_rep, mask_tokens)

        atom_loss = self.cal_atom_loss(
            pred_node=enc_rep,
            target_atom=x,
            mask_tokens=mask_tokens,
            loss_fn=self.atom_recon_type,
            alpha_l=self.alpha_l,
        )

        pe_loss_mode = getattr(self.args, "pe_loss_mode", "balanced")

        if pe_loss_mode in ["chunked", "full_chunked", "chunked_weighted"]:
            pe_loss = self.cal_pe_loss_chunked_weighted(
                pe=pe,
                target_dis=PE,
                edge_index_pe=edge_index_pe,
                mask_tokens=mask_tokens,
                num_nodes=x.size(0),
            )
        else:
            edge_index_pe_loss, PE_loss, pe_loss_input = self.select_pe_edges_for_loss(
                edge_index_pe=edge_index_pe,
                target_dis=PE,
                pe=pe,
            )

            reconstruct_dist, _ = self.pe_reconstruct_heads(
                pe_loss_input,
                edge_index_pe_loss,
            )

            pe_loss = self.cal_pe_loss_weighted(
                reconstruct_dis=reconstruct_dist,
                target_dis=PE_loss,
                edge_index_pe=edge_index_pe_loss,
                mask_tokens=mask_tokens,
                num_nodes=x.size(0),
            )

        loss = self.masked_atom_loss * atom_loss + self.masked_pe_loss * pe_loss
        return loss

    def encoding_mask_noise(self, x, u, mask_ratio, replace_ratio):
        mask_token_ratio = 1 - replace_ratio
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        if replace_ratio > 0:
            num_noise_nodes = int(replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(mask_token_ratio * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        u_masked = None

        u_masked = u.clone()
        current_noise = self._current_noise_val()
        if self.learnable_noise:
            pos_noise = (torch.rand((len(mask_nodes), u.size(1)), device=u_masked.device) * 2.0 - 1.0) * current_noise
        else:
            pos_noise = noise_fn(float(current_noise), len(mask_nodes), u.size(1)).to(u_masked.device)
        u_masked[mask_nodes] += pos_noise

        return out_x, u_masked, mask_nodes
    

    def embed(self, x, edge_index, PE):        
        enc_rep, _ = self.encoder.embed(x, edge_index, PE=PE)        
        return enc_rep



    def select_pe_edges_for_loss(self, edge_index_pe, target_dis, pe):
        """
        Memory-safe PE edge selection.
        Modes:
        - balanced: distance-balanced PE edge sampling
        - random: uniform random PE edge sampling
        - all: use all non-self-loop PE edges
        """
        mode = getattr(self.args, "pe_loss_mode", "balanced")
        max_pe_edges = int(getattr(self.args, "max_pe_edges", 500000))
        balance_bins = int(getattr(self.args, "pe_balance_bins", 5))

        src = edge_index_pe[0]
        dst = edge_index_pe[1]
        valid_mask = src != dst
        valid_idx = torch.where(valid_mask)[0]
        total_valid = valid_idx.numel()

        if max_pe_edges <= 0 or total_valid <= max_pe_edges or mode == "all":
            if not hasattr(self, "_pe_select_printed"):
                print(f"[PE-select] using all {total_valid}/{edge_index_pe.size(1)} non-self PE edges")
                self._pe_select_printed = True
            selected = valid_idx
            return edge_index_pe[:, selected], target_dis[selected], pe[selected]

        if mode == "random":
            perm = torch.randperm(total_valid, device=edge_index_pe.device)[:max_pe_edges]
            selected = valid_idx[perm]

            if not hasattr(self, "_pe_select_printed"):
                print(f"[PE-random-fix] sampled {selected.numel()}/{total_valid} non-self PE edges")
                self._pe_select_printed = True

            return edge_index_pe[:, selected], target_dis[selected], pe[selected]

        # Default: balanced distance-based sampling.
        # Build bin cache once because target PE distances are fixed for a dataset.
        cache_ok = (
            hasattr(self, "_pe_balance_bins_cache")
            and hasattr(self, "_pe_balance_cache_total")
            and self._pe_balance_cache_total == total_valid
            and hasattr(self, "_pe_balance_cache_bins")
            and self._pe_balance_cache_bins == balance_bins
        )

        if not cache_ok:
            dis = target_dis[valid_idx].detach()
            qs = torch.quantile(
                dis,
                torch.linspace(0, 1, balance_bins + 1, device=dis.device),
            )

            bins = []
            for i in range(balance_bins):
                if i == 0:
                    m = dis <= qs[i + 1]
                elif i == balance_bins - 1:
                    m = dis > qs[i]
                else:
                    m = (dis > qs[i]) & (dis <= qs[i + 1])

                bin_idx = valid_idx[torch.where(m)[0]]
                if bin_idx.numel() > 0:
                    bins.append(bin_idx)

            self._pe_balance_bins_cache = bins
            self._pe_balance_cache_total = total_valid
            self._pe_balance_cache_bins = balance_bins

        bins = self._pe_balance_bins_cache
        per_bin = max(1, max_pe_edges // max(1, len(bins)))

        chosen = []
        for bin_idx in bins:
            n = bin_idx.numel()
            if n <= per_bin:
                chosen.append(bin_idx)
            else:
                perm = torch.randperm(n, device=edge_index_pe.device)[:per_bin]
                chosen.append(bin_idx[perm])

        selected = torch.cat(chosen, dim=0)

        if selected.numel() < max_pe_edges:
            remain = max_pe_edges - selected.numel()
            extra = valid_idx[torch.randint(0, total_valid, (remain,), device=edge_index_pe.device)]
            selected = torch.cat([selected, extra], dim=0)

        if selected.numel() > max_pe_edges:
            perm = torch.randperm(selected.numel(), device=edge_index_pe.device)[:max_pe_edges]
            selected = selected[perm]

        selected = selected[torch.randperm(selected.numel(), device=edge_index_pe.device)]

        if not hasattr(self, "_pe_select_printed"):
            print(
                f"[PE-balanced-fix] sampled {selected.numel()}/{total_valid} "
                f"non-self PE edges across {len(bins)} distance bins"
            )
            self._pe_select_printed = True

        return edge_index_pe[:, selected], target_dis[selected], pe[selected]


    def cal_pe_loss_weighted(self, reconstruct_dis, target_dis, edge_index_pe, mask_tokens, num_nodes):
        """
        Weighted PE reconstruction loss.
        Higher PE-distance edges can receive slightly higher weights.
        """
        edge_index, target_dis = remove_self_loops(
            edge_index=edge_index_pe,
            edge_attr=target_dis,
        )
        edge_index, target_dis = to_undirected(
            edge_index=edge_index,
            edge_attr=target_dis,
            reduce="mean",
        )

        reconstruct_dis = reconstruct_dis.view(-1)
        target_dis = target_dis.view(-1)

        row = edge_index[0]
        mask_lookup = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index_pe.device)
        mask_lookup[mask_tokens] = True
        idx = mask_lookup[row]

        if not idx.any():
            return reconstruct_dis.sum() * 0.0

        pred = reconstruct_dis[idx]
        target = target_dis[idx]

        loss_vec = F.smooth_l1_loss(
            pred,
            target,
            reduction="none",
            beta=1.0,
        )

        weight_mode = getattr(self.args, "pe_weight_mode", "distance")
        beta = float(getattr(self.args, "pe_weight_beta", 1.0))
        clip_min = float(getattr(self.args, "pe_weight_clip_min", 0.5))
        clip_max = float(getattr(self.args, "pe_weight_clip_max", 2.0))

        if weight_mode == "distance":
            denom = target.detach().mean().clamp_min(1e-6)
            weights = (target.detach() / denom).clamp(min=clip_min, max=clip_max)
            weights = 1.0 + beta * (weights - 1.0)
            weights = weights.clamp(min=clip_min, max=clip_max)
            return (loss_vec * weights).sum() / weights.sum().clamp_min(1e-6)

        return loss_vec.mean()


    def cal_pe_loss_chunked_weighted(self, pe, target_dis, edge_index_pe, mask_tokens, num_nodes):
        """
        Full PE-loss mode.
        Uses all PE edges but computes loss in chunks.
        Safer than full one-shot loss, but slower than balanced sampling.
        """
        chunk_size = int(getattr(self.args, "pe_loss_chunk_size", 100000))
        total_edges = edge_index_pe.size(1)

        if not hasattr(self, "_pe_chunk_printed"):
            print(f"[PE-chunk-fix] using all {total_edges} PE edges in chunks of {chunk_size}")
            self._pe_chunk_printed = True

        loss_sum = pe.new_tensor(0.0)
        weight_sum = pe.new_tensor(0.0)

        for start in range(0, total_edges, chunk_size):
            end = min(start + chunk_size, total_edges)

            edge_chunk = edge_index_pe[:, start:end]
            pe_chunk = pe[start:end]
            target_chunk = target_dis[start:end]

            reconstruct_dist, _ = self.pe_reconstruct_heads(pe_chunk, edge_chunk)

            cur_loss = self.cal_pe_loss_weighted(
                reconstruct_dis=reconstruct_dist,
                target_dis=target_chunk,
                edge_index_pe=edge_chunk,
                mask_tokens=mask_tokens,
                num_nodes=num_nodes,
            )

            cur_edges = edge_chunk.size(1)
            loss_sum = loss_sum + cur_loss * cur_edges
            weight_sum = weight_sum + cur_edges

        return loss_sum / weight_sum.clamp_min(1.0)


    def cal_pe_loss(self, reconstruct_dis, target_dis, edge_index_pe, mask_tokens):
        edge_index, target_dis = remove_self_loops(edge_index=edge_index_pe, edge_attr=target_dis)
        edge_index, target_dis = to_undirected(edge_index=edge_index, edge_attr=target_dis, reduce="mean")
        row = edge_index[0]
        idx = torch.isin(row, mask_tokens)

        reconstruct_dis = reconstruct_dis[idx]
        target_dis = target_dis[idx]

        pe_reconstruct_loss = F.smooth_l1_loss(
            reconstruct_dis,
            target_dis,
            reduction="mean",
            beta=1.0,
        )

        return pe_reconstruct_loss

    def cal_atom_loss(self, pred_node, target_atom, mask_tokens, loss_fn, alpha_l=0.0):
        target_atom = target_atom[mask_tokens]

        if loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
            atom_loss = criterion(pred_node, target_atom)
        elif loss_fn == "mse":
            atom_loss = self.mse_loss(pred_node, target_atom)
        else:
            criterion = nn.CrossEntropyLoss()
            atom_loss = criterion(pred_node, target_atom)

        return atom_loss

    def sce_loss(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def mse_loss(self, x, y):
        loss = ((x - y) ** 2).mean()
        return loss
