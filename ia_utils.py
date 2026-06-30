import torch

_PRINTED = set()


def _minmax(x, eps=1e-12):
    return (x - x.min()) / (x.max() - x.min() + eps)


def _safe_get(args, name, default):
    if args is None:
        return default
    return getattr(args, name, default)


def _all_self_loops(num_nodes, device):
    nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    return torch.stack([nodes, nodes], dim=0)


def _balanced_select_by_spectral_bins(spectral, degree_score, budget, bins):
    """
    Balanced IA selection:
    1. Sort edges by spectral distance.
    2. Divide into quantile-like bins.
    3. Select important edges from every bin.
    4. Importance inside each bin = spectral + degree.
    """
    M = spectral.numel()
    device = spectral.device

    bins = int(max(1, min(bins, budget, M)))
    order = torch.argsort(spectral)

    combined = 0.6 * _minmax(spectral) + 0.4 * _minmax(degree_score)

    selected_parts = []
    selected_mask = torch.zeros(M, dtype=torch.bool, device=device)

    base = budget // bins
    rem = budget % bins

    for b in range(bins):
        start = (b * M) // bins
        end = ((b + 1) * M) // bins
        cand = order[start:end]

        if cand.numel() == 0:
            continue

        k = base + (1 if b < rem else 0)
        k = min(k, cand.numel())

        if k <= 0:
            continue

        local_scores = combined[cand]
        local_top = torch.topk(local_scores, k=k, largest=True).indices
        chosen = cand[local_top]

        selected_parts.append(chosen)
        selected_mask[chosen] = True

    if selected_parts:
        selected = torch.cat(selected_parts, dim=0)
    else:
        selected = torch.empty(0, dtype=torch.long, device=device)

    if selected.numel() < budget:
        remaining = torch.where(~selected_mask)[0]
        need = budget - selected.numel()

        if remaining.numel() > 0:
            k = min(need, remaining.numel())
            fill_top = torch.topk(combined[remaining], k=k, largest=True).indices
            fill = remaining[fill_top]
            selected = torch.cat([selected, fill], dim=0)

    if selected.numel() > budget:
        selected = selected[:budget]

    return selected


def select_budgeted_pe_edges(edge_index_pe, u, graph_edge_index=None, args=None, device=None):
    """
    Self-loop-safe budgeted PE-edge selector for Budgeted GraphPAE.

    Modes:
      full      : all PE edges
      random    : random non-self PE edges + all self-loops
      spectral  : top spectral-distance non-self PE edges + all self-loops
      degree    : top endpoint-degree non-self PE edges + all self-loops
      ia        : balanced spectral-bin + degree-aware non-self PE edges + all self-loops

    Main paper method:
      ia = Balanced Importance-Aware Positional Edge Selection.
    """

    if edge_index_pe is None:
        return edge_index_pe

    mode = str(_safe_get(args, "pe_sampler", "full")).lower()

    if mode in ["full", "all", "none", "no", "false"]:
        return edge_index_pe

    dev = edge_index_pe.device
    num_nodes = int(u.size(0))

    src_all = edge_index_pe[0]
    dst_all = edge_index_pe[1]

    nonself_mask = src_all != dst_all
    nonself_edges = edge_index_pe[:, nonself_mask]

    M_nonself = int(nonself_edges.size(1))

    if M_nonself <= 1:
        return _all_self_loops(num_nodes, dev)

    ratio = float(_safe_get(args, "pe_budget_ratio", 0.0) or 0.0)
    max_pe_edges = int(_safe_get(args, "max_pe_edges", 0) or 0)

    if ratio > 0:
        budget_nonself = max(1, int(M_nonself * ratio))
    elif max_pe_edges > 0:
        budget_nonself = max_pe_edges
    else:
        return edge_index_pe

    budget_nonself = min(int(budget_nonself), M_nonself)

    src = nonself_edges[0]
    dst = nonself_edges[1]

    if mode == "random":
        idx = torch.randperm(M_nonself, device=dev)[:budget_nonself]

    else:
        u_work = u.to(dev) if u.device != dev else u

        max_freqs = int(_safe_get(args, "max_freqs", u_work.size(1)) or u_work.size(1))
        score_freqs = int(_safe_get(args, "ia_max_freqs_for_score", max_freqs) or max_freqs)
        score_freqs = max(1, min(score_freqs, u_work.size(1)))

        U = u_work[:, :score_freqs]

        spectral = torch.norm(U[src] - U[dst], p=2, dim=1)
        spectral = _minmax(spectral)

        deg = torch.zeros(num_nodes, device=dev)

        if graph_edge_index is not None:
            ge = graph_edge_index.to(dev)
            one = torch.ones(ge.size(1), device=dev)
            deg.index_add_(0, ge[0], one)
            deg.index_add_(0, ge[1], one)

        degree_score = torch.log1p(deg[src]) + torch.log1p(deg[dst])
        degree_score = _minmax(degree_score)

        if mode == "spectral":
            idx = torch.topk(spectral, k=budget_nonself, largest=True).indices

        elif mode == "degree":
            idx = torch.topk(degree_score, k=budget_nonself, largest=True).indices

        elif mode in ["ia", "importance", "balanced", "ia_balanced"]:
            bins = int(_safe_get(args, "ia_bins", 10) or 10)
            idx = _balanced_select_by_spectral_bins(
                spectral=spectral,
                degree_score=degree_score,
                budget=budget_nonself,
                bins=bins,
            )

        else:
            idx = torch.randperm(M_nonself, device=dev)[:budget_nonself]

    selected_nonself = nonself_edges[:, idx]
    self_edges = _all_self_loops(num_nodes, dev)

    selected = torch.cat([selected_nonself, self_edges], dim=1)

    key = (mode, M_nonself, budget_nonself, num_nodes)
    if key not in _PRINTED:
        _PRINTED.add(key)
        print(
            f"[IA-Budget] sampler={mode}, "
            f"nonself={budget_nonself}/{M_nonself} ({100.0 * budget_nonself / M_nonself:.2f}%), "
            f"self_loops={num_nodes}, total_active_edges={selected.size(1)}"
        )

    return selected
