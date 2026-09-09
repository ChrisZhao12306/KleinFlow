"""Structure-conditioned graph training, Flow Matching, and constrained decoding."""
from flow_klein.paths import OUTPUT_ROOT

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from flow_klein.data.structural import Datasets, list_graph_loader, data_split, BFS, structural_BFS
from flow_klein.models.kernels import kernel
from flow_klein.models.structural import KleinEncoder, KleinGraphVAE, MaskedGraphDecoder
from flow_klein.models.flow_matching import KleinFlowMatching
from flow_klein.geometry.klein import Klein
from flow_klein.evaluation.spectre import degree_stats, clustering_stats, spectral_stats
from flow_klein.data.benchmarks_structural import (
    is_benchmark_dataset,
    load_benchmark_splits,
    normalize_benchmark_name,
)
from flow_klein.evaluation.structural import (
    compute_vun,
    evaluate_external_benchmark,
    evaluation_profile,
    is_planar_graph,
    is_tree_graph,
    normalize_graph,
    preflight_metric_dependencies,
)
from flow_klein.data.profile import DatasetProfile


OUTER_SPLIT_SEED = 123
USES_INNER_VALIDATION_SPLIT = True




def _write_metrics_json(args, results, graph_save_path, timestamp):
    from flow_klein.training.reporting import write_metrics_json
    write_metrics_json(args, results, graph_save_path, timestamp,
                       OUTER_SPLIT_SEED, USES_INNER_VALIDATION_SPLIT)



def load_data(args):
    """Prepare dataset-specific training, validation, and test graph collections.
    
    Returns the training dataset, test dataset, validation adjacencies, test
    adjacencies, and training adjacencies. Benchmark datasets use predefined
    splits. Grid reserves a deterministic validation subset before fitting
    features and training-set structural profiles."""
    dataset = args.dataset
    dataset_kwargs = {
        "node_feat_mode": getattr(args, "node_feat_mode", "struct"),
        "lap_pe_dim": getattr(args, "lap_pe_dim", 8),
    }

    self_for_none = True

    if is_benchmark_dataset(dataset):
        canonical = normalize_benchmark_name(dataset)
        args.dataset = canonical
        splits = load_benchmark_splits(canonical)
        train_core_adj = list(splits["train"])
        val_adj = list(splits["val"])
        test_list_adj = list(splits["test"])
        benchmark_ordering = str(
            getattr(args, "benchmark_ordering", "none")
        ).strip().lower()
        if benchmark_ordering == "structural_bfs":
            train_core_adj = structural_BFS(train_core_adj)
            val_adj = structural_BFS(val_adj)
            test_list_adj = structural_BFS(test_list_adj)
        elif benchmark_ordering != "none":
            raise ValueError(
                "benchmark_ordering must be 'none' or 'structural_bfs', "
                f"got {benchmark_ordering!r}"
            )
        elif args.bfsOrdering:
            train_core_adj = BFS(train_core_adj)
            val_adj = BFS(val_adj)
            test_list_adj = BFS(test_list_adj)

        all_adj = train_core_adj + val_adj + test_list_adj
        max_size = max(adj.shape[0] for adj in all_adj)
        train_x = [None for _ in train_core_adj]
        val_x = [None for _ in val_adj]
        test_x = [None for _ in test_list_adj]
        list_graphs = Datasets(
            train_core_adj, self_for_none, train_x, None,
            Max_num=max_size, set_diag_of_isol_Zer=False, **dataset_kwargs
        )
        val_graphs = Datasets(
            val_adj, self_for_none, val_x, None,
            Max_num=max_size, set_diag_of_isol_Zer=False, **dataset_kwargs
        )
        list_test_graphs = Datasets(
            test_list_adj, self_for_none, test_x, None,
            Max_num=max_size, set_diag_of_isol_Zer=False, **dataset_kwargs
        )
        print(
            f"[load_data] fixed benchmark split: train={len(train_core_adj)}  "
            f"val={len(val_adj)}  test={len(test_list_adj)}"
        )
        return (
            list_graphs,
            list_test_graphs,
            val_adj,
            test_list_adj,
            train_core_adj,
            val_graphs,
        )

    list_adj, list_x, list_label = list_graph_loader(
        dataset, return_labels=True, shuffle=False
    )

    if args.bfsOrdering:
        list_adj = BFS(list_adj)

    if len(list_adj) == 1:
        test_list_adj = list_adj.copy()
        list_graphs = Datasets(list_adj, self_for_none, list_x, None, **dataset_kwargs)
        list_test_graphs = list_graphs
        val_adj = list_adj
        return list_graphs, list_test_graphs, val_adj, test_list_adj, list_adj, None

    # 80/20 train/test split, frozen to the seed used by the search runs.
    max_size = None
    list_adj_tv, test_list_adj, list_x_tv, list_x_test, _, list_label_test = data_split(
        list_adj, list_x, list_label, split_seed=OUTER_SPLIT_SEED
    )

    # Carve disjoint validation slice off the train+val pool.
    n_tv = len(list_adj_tv)
    val_ratio = float(getattr(args, "val_ratio", 0.10))
    n_val = max(1, int(round(val_ratio * n_tv))) if n_tv > 1 else 0

    rng = np.random.RandomState(int(getattr(args, "split_seed", 1432)))
    all_idx = np.arange(n_tv)
    rng.shuffle(all_idx)
    val_idx = sorted(all_idx[:n_val].tolist())
    train_idx = sorted(all_idx[n_val:].tolist())

    if getattr(args, "assert_disjoint_val", True):
        assert set(train_idx).isdisjoint(set(val_idx)), \
            "Training and validation indices must be disjoint."

    train_core_adj = [list_adj_tv[i] for i in train_idx]
    val_adj = [list_adj_tv[i] for i in val_idx] if n_val > 0 else []
    list_x_train = [list_x_tv[i] for i in train_idx] if list_x_tv is not None else None
    list_label_train = None  # Graph generation does not require class labels.

    list_graphs = Datasets(
        train_core_adj, self_for_none, list_x_train, list_label_train,
        Max_num=max_size, set_diag_of_isol_Zer=False, **dataset_kwargs
    )
    list_test_graphs = Datasets(
        test_list_adj, self_for_none, list_x_test, list_label_test,
        Max_num=list_graphs.max_num_nodes, set_diag_of_isol_Zer=False, **dataset_kwargs
    )
    # val_graphs uses the same max_num_nodes so the encoder shape is consistent.
    if n_val > 0:
        list_x_val = [list_x_tv[i] for i in val_idx] if list_x_tv is not None else None
        val_graphs = Datasets(
            val_adj, self_for_none, list_x_val, None,
            Max_num=list_graphs.max_num_nodes, set_diag_of_isol_Zer=False, **dataset_kwargs
        )
    else:
        val_graphs = None

    print(f"[load_data] train_core={len(train_core_adj)}  val={len(val_adj)}  test={len(test_list_adj)}")

    return list_graphs, list_test_graphs, val_adj, test_list_adj, train_core_adj, val_graphs


def get_subGraph_features(args, org_adj, subgraphs_indexes, kernel_model):
    """Extract dense subgraph tensors and optional kernel features."""
    device = args.device
    subgraphs = []
    target_kernel_val = None

    for i in range(len(org_adj)):
        subGraph = org_adj[i]
        if subgraphs_indexes is not None:
            subGraph = subGraph[:, subgraphs_indexes[i]]
            subGraph = subGraph[subgraphs_indexes[i], :]
        subGraph = torch.tensor(subGraph.todense(), dtype=torch.float32)
        subgraphs.append(subGraph)

    subgraphs = torch.stack(subgraphs).to(device)
    if kernel_model is not None:
        target_kernel_val = kernel_model(subgraphs)
        target_kernel_val = [val.to("cpu") for val in target_kernel_val]

    subgraphs = subgraphs.to("cpu")
    torch.cuda.empty_cache()
    return target_kernel_val, subgraphs


def build_node_mask(node_counts, max_nodes, device):
    """Build a binary node-validity mask from graph sizes."""
    counts = torch.tensor(node_counts, device=device, dtype=torch.long)
    arange = torch.arange(max_nodes, device=device).unsqueeze(0)
    return (arange < counts.unsqueeze(1)).float()


def compute_graph_batch_stats(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Compute graph-level targets on padded adjacency batches.

    Returns a 6-D vector for backward compatibility with the existing
    `stats_pred` and `encoder_aux` heads:
      [node_count, edge_count, density, avg_degree, deg_mom2, deg_mom3]
    """
    adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_idx = torch.arange(adj.shape[-1], device=adj.device)
    adj = adj.clone()
    adj[:, diag_idx, diag_idx] = 0.0

    node_count = node_mask.sum(dim=1)
    edge_count = adj.sum(dim=(1, 2)) / 2.0
    max_edges = (node_count * (node_count - 1.0) / 2.0).clamp_min(1.0)
    density = edge_count / max_edges
    degrees = adj.sum(dim=-1)
    avg_degree = (degrees * node_mask).sum(dim=1) / node_count.clamp_min(1.0)
    degree_sq_mean = (degrees.pow(2) * node_mask).sum(dim=1) / node_count.clamp_min(1.0)
    degree_cube_mean = (degrees.pow(3) * node_mask).sum(dim=1) / node_count.clamp_min(1.0)
    return torch.stack(
        [node_count, edge_count, density, avg_degree, degree_sq_mean, degree_cube_mean],
        dim=-1
    )


def compute_profile_scalar_stats(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Compute the 8-D scalar vector used by DatasetProfile per batch.

    Matches klein_dataset_configs._scalar_stats_from_dense:
      [n, m, density, avg_deg, deg_mom2, deg_mom3, deg_mom4, max_deg]
    """
    adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_idx = torch.arange(adj.shape[-1], device=adj.device)
    adj = adj.clone()
    adj[:, diag_idx, diag_idx] = 0.0

    n = node_mask.sum(dim=1)
    edges = adj.sum(dim=(1, 2)) / 2.0
    max_edges = (n * (n - 1.0) / 2.0).clamp_min(1.0)
    density = edges / max_edges
    deg = adj.sum(dim=-1) * node_mask
    avg_deg = deg.sum(dim=1) / n.clamp_min(1.0)
    deg_m2 = (deg.pow(2) * node_mask).sum(dim=1) / n.clamp_min(1.0)
    deg_m3 = (deg.pow(3) * node_mask).sum(dim=1) / n.clamp_min(1.0)
    deg_m4 = (deg.pow(4) * node_mask).sum(dim=1) / n.clamp_min(1.0)
    max_deg = deg.amax(dim=1)
    return torch.stack(
        [n, edges, density, avg_deg, deg_m2, deg_m3, deg_m4, max_deg], dim=-1
    )


def compute_soft_degree_histogram(
    pair_probs: torch.Tensor,
    pair_mask: torch.Tensor,
    node_mask: torch.Tensor,
    max_degree: float,
    n_bins: int = 32,
    bandwidth: float = None,
) -> torch.Tensor:
    """Differentiable soft degree histogram.
    
    Args:
        pair_probs: (B, N, N) sigmoid probabilities (already pair-masked).
        pair_mask: (B, N, N) pairwise validity mask with diagonal zeroed.
        node_mask: (B, N) per-node validity.
        max_degree: dataset-aware upper bound for the degree axis. Bins are
            placed linearly on `[0, max_degree]`, **same scale for predicted
            and target histograms** so the L1 distance is meaningful.
        n_bins: number of histogram bins.
        bandwidth: Gaussian bin width. When None, defaults to
            `0.75 * (max_degree / (n_bins - 1))` so adjacent bins overlap.
    
    Returns:
        (B, n_bins) row-normalized soft histograms.
    
    Predicted and target histograms share the same degree coordinates,
    so their L1 distance measures distribution overlap on a common scale."""
    masked_probs = pair_probs * pair_mask
    soft_deg = masked_probs.sum(dim=-1)  # (B, N), natural degree scale

    max_degree = max(float(max_degree), 1.0)
    if bandwidth is None:
        bandwidth = 0.75 * (max_degree / max(n_bins - 1, 1))
    bandwidth = max(float(bandwidth), 1e-3)

    centers = torch.linspace(0.0, max_degree, n_bins,
                             device=soft_deg.device, dtype=soft_deg.dtype)

    diff = soft_deg.unsqueeze(-1) - centers.view(1, 1, -1)  # (B, N, bins)
    weights = torch.exp(-0.5 * (diff / bandwidth) ** 2)
    weights = weights * node_mask.unsqueeze(-1)
    hist = weights.sum(dim=1)  # (B, bins)
    hist = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return hist


def calibrate_edge_probabilities_to_budget(
    adj_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    target_edges: torch.Tensor,
    temperature: float = 0.5,
    bisection_steps: int = 40,
) -> torch.Tensor:
    """Return symmetric soft adjacencies with the requested expected edge count.
    
    A detached per-graph threshold is found by bisection to calibrate the
    weighted-BCE logits. Gradients flow through the shifted logits. This
    calibration is enabled by the constrained benchmark configurations."""
    if adj_logits.ndim != 3 or adj_logits.shape[-1] != adj_logits.shape[-2]:
        raise ValueError(f"Expected square batched logits, got {adj_logits.shape}")
    temperature = max(float(temperature), 1e-3)
    batch_size, node_count, _ = adj_logits.shape
    upper = torch.triu(
        torch.ones(node_count, node_count, device=adj_logits.device, dtype=adj_logits.dtype),
        diagonal=1,
    )
    calibrated = []
    for batch_index in range(batch_size):
        valid_upper = upper * pair_mask[batch_index]
        valid = valid_upper.bool()
        values = adj_logits[batch_index][valid]
        if values.numel() == 0:
            calibrated.append(torch.zeros_like(adj_logits[batch_index]))
            continue
        edge_budget = float(
            torch.clamp(
                target_edges[batch_index].detach(),
                0,
                values.numel(),
            ).item()
        )
        with torch.no_grad():
            low = values.min() - 20.0 * temperature
            high = values.max() + 20.0 * temperature
            for _ in range(int(bisection_steps)):
                midpoint = 0.5 * (low + high)
                expected = torch.sigmoid((values - midpoint) / temperature).sum()
                if float(expected.item()) > edge_budget:
                    low = midpoint
                else:
                    high = midpoint
            threshold = 0.5 * (low + high)
        upper_probs = (
            torch.sigmoid((adj_logits[batch_index] - threshold) / temperature)
            * valid_upper
        )
        calibrated.append(upper_probs + upper_probs.transpose(0, 1))
    return torch.stack(calibrated, dim=0)


def compute_kernel_matching_loss(reconstructed_adj, target_kernel_val, kernel_model, alpha=0.1):
    """Auxiliary kernel matching loss with reduced weight."""
    if kernel_model is None or target_kernel_val is None:
        return reconstructed_adj.new_tensor(0.0)
    reconstructed_kernel_val = kernel_model(reconstructed_adj)
    kernel_loss = reconstructed_adj.new_tensor(0.0)
    for generated, target in zip(reconstructed_kernel_val, target_kernel_val):
        kernel_loss = kernel_loss + F.smooth_l1_loss(generated, target.to(generated.device))
    return alpha * kernel_loss


def compute_vae_loss(
    adj_logits,
    adj_probs,
    target_adj,
    node_logits,
    node_mask,
    stats_pred,
    stats_target,
    encoder_aux,
    log_std,
    mean,
    kernel_model,
    target_kernel_val,
    degree_pred=None,
    kernel_weight=0.1,
    node_weight=0.25,
    stats_weight=0.1,
    kl_weight=0.05,
    degree_reg_weight=0.10,
    degree_aux_weight=0.05,
    degree_hist_max=None,
    degree_hist_budget_calibration=False,
    degree_hist_temperature=0.5,
):
    """Compute masked reconstruction and auxiliary graph-structure losses.
    
    The loss includes node and edge reconstruction, graph-level statistics,
    latent regularization, per-node Smooth-L1 degree regression, and an L1
    penalty on differentiable degree histograms, weighted by the configuration."""
    diag_idx = torch.arange(target_adj.shape[-1], device=target_adj.device)
    target_adj = target_adj.clone().float()
    target_adj[:, diag_idx, diag_idx] = 0.0

    pair_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    pair_mask[:, diag_idx, diag_idx] = 0.0

    positive = (target_adj * pair_mask).sum().clamp_min(1.0)
    negative = pair_mask.sum().clamp_min(1.0) - positive
    pos_weight = (negative / positive).clamp_min(1.0)

    edge_loss_raw = F.binary_cross_entropy_with_logits(
        adj_logits.float(),
        target_adj,
        reduction="none",
        pos_weight=pos_weight
    )
    edge_loss = (edge_loss_raw * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    node_loss = F.binary_cross_entropy_with_logits(node_logits.float(), node_mask)
    stats_loss = F.smooth_l1_loss(stats_pred, stats_target) + F.smooth_l1_loss(encoder_aux, stats_target)

    norm_kl = mean.shape[0] * mean.shape[1]
    kl_loss = (1 / norm_kl) * -0.5 * torch.sum(
        1 + 2 * log_std - mean.pow(2) - torch.exp(log_std).pow(2)
    )

    kernel_loss = compute_kernel_matching_loss(
        reconstructed_adj=adj_probs,
        target_kernel_val=target_kernel_val,
        kernel_model=kernel_model,
        alpha=kernel_weight,
    )

    # -------- Degree regression --------
    if degree_pred is not None and degree_reg_weight > 0:
        with torch.no_grad():
            target_deg = (target_adj * pair_mask).sum(dim=-1).float()
        mask_flat = node_mask.bool()
        if mask_flat.any():
            degree_reg_loss = F.smooth_l1_loss(
                degree_pred[mask_flat],
                target_deg[mask_flat],
            )
        else:
            degree_reg_loss = adj_logits.new_tensor(0.0)
    else:
        degree_reg_loss = adj_logits.new_tensor(0.0)

    # -------- Soft degree histogram --------
    if degree_aux_weight > 0:
        # Bin scale is dataset-aware: callers pass `degree_hist_max` from
        # DatasetProfile (mean max_degree + 2 std). Fallback to a per-batch
        # estimate so the function still works when no profile is available.
        if degree_hist_max is None or degree_hist_max <= 0:
            with torch.no_grad():
                batch_max_deg = (target_adj * pair_mask).sum(dim=-1).max().item()
            degree_hist_max = max(batch_max_deg, 1.0)
        degree_probs = adj_probs
        if degree_hist_budget_calibration:
            degree_probs = calibrate_edge_probabilities_to_budget(
                adj_logits,
                pair_mask,
                stats_target[:, 1],
                temperature=degree_hist_temperature,
            )
        soft_hist = compute_soft_degree_histogram(
            degree_probs, pair_mask, node_mask, max_degree=degree_hist_max,
        )
        with torch.no_grad():
            target_hist = compute_soft_degree_histogram(
                target_adj, pair_mask, node_mask, max_degree=degree_hist_max,
            )
        degree_hist_loss = (soft_hist - target_hist).abs().sum(dim=-1).mean()
    else:
        degree_hist_loss = adj_logits.new_tensor(0.0)

    total_loss = (
        edge_loss
        + node_weight * node_loss
        + stats_weight * stats_loss
        + kl_weight * kl_loss
        + kernel_loss
        + degree_reg_weight * degree_reg_loss
        + degree_aux_weight * degree_hist_loss
    )

    predictions = (adj_probs > 0.5).float()
    acc = ((predictions == target_adj).float() * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    return {
        "total": total_loss,
        "edge": edge_loss,
        "node": node_loss,
        "stats": stats_loss,
        "kl": kl_loss,
        "kernel": kernel_loss,
        "degree_reg": degree_reg_loss,
        "degree_hist": degree_hist_loss,
        "acc": acc,
    }


def prepare_batch_graphs(org_adj, device):
    """Create a batched DGL graph from scipy adjacency matrices."""
    graphs = []
    for graph in org_adj:
        graph = graph.copy()
        graph.setdiag(1)
        graphs.append(dgl.from_scipy(graph))
    return dgl.batch(graphs).to(device)


@torch.no_grad()
def collect_encoder_outputs(args, model, list_graphs, device, profile=None):
    """Collect deterministic Klein embeddings and conditional codes.

    If `profile` is provided, the encoder receives the per-graph profile vector
    so the returned cond_codes include the structural conditioning component.
    """
    mini_batch_size = args.batchSize
    self_for_none = True
    model.eval()

    all_embeddings = []
    all_cond_codes = []

    for batch_idx in range(0, len(list_graphs.list_adjs), mini_batch_size):
        from_ = batch_idx
        to_ = min(from_ + mini_batch_size, len(list_graphs.list_adjs))

        org_adj, x_s, node_num_list, _, _, _ = list_graphs.get__(from_, to_, self_for_none, bfs=None)
        if len(org_adj) == 0:
            continue

        x_s = torch.cat(x_s).reshape(-1, x_s[0].shape[-1]).to(device)
        batch_size = [len(org_adj), org_adj[0].shape[0]]
        node_mask = build_node_mask(node_num_list, batch_size[1], device)
        org_adj_dgl = prepare_batch_graphs(org_adj, device)

        profile_vec = None
        if profile is not None and getattr(model.encoder, "use_struct_cond", False):
            profile_vec = compute_profile_vec_from_adj(org_adj, node_mask, profile, device)

        h_klein, _, _, cond_code, _ = model.encode(
            org_adj_dgl, x_s, batch_size, node_mask=node_mask,
            profile_vec=profile_vec,
        )
        all_embeddings.append(h_klein.cpu())
        all_cond_codes.append(cond_code.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_cond_codes, dim=0)


def compute_profile_vec_from_adj(org_adj, node_mask, profile, device):
    """Build a per-graph profile vector consumable by the encoder.

    Concatenates [standardized 8-D scalars, raw 32-bin degree histogram].
    """
    _, subgraphs = get_subGraph_features(_ArgsLike(device), org_adj, None, None)
    subgraphs = subgraphs.to(device)
    raw_scalar = compute_profile_scalar_stats(subgraphs, node_mask)  # (B, 8)
    std_scalar = profile.standardize_scalar(raw_scalar)
    hist = _batch_degree_histogram(subgraphs, node_mask, profile.n_bins)
    return torch.cat([std_scalar, hist], dim=-1)


class _ArgsLike:
    """Tiny shim so we can reuse `get_subGraph_features` without args."""
    def __init__(self, device):
        self.device = device


def _batch_degree_histogram(adj: torch.Tensor, node_mask: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Hard (non-differentiable) degree histogram per graph; used to build the
    encoder-side profile vector at train and validation time. Matches the bin
    edges produced by `klein_dataset_configs._degree_histogram`."""
    diag_idx = torch.arange(adj.shape[-1], device=adj.device)
    adj_masked = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    adj_masked = adj_masked.clone()
    adj_masked[:, diag_idx, diag_idx] = 0.0
    deg = adj_masked.sum(dim=-1) * node_mask  # (B, N)

    n_valid = node_mask.sum(dim=1).clamp_min(1.0)  # (B,)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=adj.device)  # normalized
    # Per-row max possible degree:
    max_deg_per_row = torch.clamp(n_valid - 1.0, min=1.0)

    hist = torch.zeros(adj.shape[0], n_bins, device=adj.device, dtype=adj.dtype)
    for i in range(adj.shape[0]):
        v = deg[i, : int(n_valid[i].item())].detach().cpu().numpy()
        if v.size == 0:
            continue
        import numpy as _np
        ed = _np.linspace(0.0, float(max_deg_per_row[i].item()), n_bins + 1)
        h, _ = _np.histogram(v, bins=ed)
        if h.sum() > 0:
            h = h / h.sum()
        hist[i] = torch.from_numpy(h).to(adj.device).to(adj.dtype)
    return hist


def _adj_to_nx(adj):
    """Convert one scipy adjacency to a clean networkx graph for MMD."""
    a = adj
    if hasattr(a, "toarray"):
        a = a.toarray()
    a = np.asarray(a)
    np.fill_diagonal(a, 0.0)
    G = nx.from_numpy_array((a > 0).astype(np.float32))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def _val_subset(val_adj, cap):
    if cap is None or cap <= 0 or cap >= len(val_adj):
        return val_adj
    return list(val_adj[:cap])


def checkpoint_selection_key(metrics, structure_constraint: str = "none"):
    """Return the lexicographic key used for validation checkpoint selection."""
    constraint = str(structure_constraint).strip().lower()
    if constraint in {"planar", "tree"}:
        vun = float(metrics.get("vun", float("-inf")))
        proxy_ratio = float(metrics.get("proxy_ratio", float("inf")))
        if not np.isfinite(vun):
            vun = float("-inf")
        if not np.isfinite(proxy_ratio) or proxy_ratio < 0:
            proxy_ratio = float("inf")
        return -vun, proxy_ratio
    avg_mmd = float(metrics.get("avg_mmd", float("inf")))
    if not np.isfinite(avg_mmd):
        avg_mmd = float("inf")
    return (avg_mmd,)


def benchmark_validation_metrics(args, real_graphs, pred_graphs, train_graphs):
    """Cheap SimGFM-aligned validation metrics for Planar/Tree checkpoints.

    V.U.N. is computed exactly.  The Ratio proxy uses only degree,
    clustering, and spectral MMD with ``compute_emd=False``; it intentionally
    omits ORCA and wavelets from periodic checkpointing.
    """
    constraint = str(getattr(args, "structure_constraint", "none")).lower()
    if constraint not in {"planar", "tree"}:
        raise ValueError(
            "benchmark_validation_metrics is only for planar/tree constraints"
        )
    if not real_graphs or not pred_graphs or not train_graphs:
        return {
            "vun": 0.0,
            "degree": float("inf"),
            "clustering": float("inf"),
            "spectral": float("inf"),
            "proxy_ratio": float("inf"),
        }
    validity_function = is_planar_graph if constraint == "planar" else is_tree_graph
    vun_metrics = compute_vun(pred_graphs, train_graphs, validity_function)
    generated_metrics = {
        "degree": float(degree_stats(real_graphs, pred_graphs, compute_emd=False)),
        "clustering": float(
            clustering_stats(real_graphs, pred_graphs, compute_emd=False)
        ),
        "spectral": float(
            spectral_stats(real_graphs, pred_graphs, compute_emd=False)
        ),
    }

    # The denominator is fixed for a run, so cache it on the argparse
    # namespace.  Underscore-prefixed fields are excluded from metrics.json.
    reference_metrics = getattr(args, "_checkpoint_proxy_reference", None)
    if reference_metrics is None:
        reference_metrics = {
            "degree": float(
                degree_stats(train_graphs, real_graphs, compute_emd=False)
            ),
            "clustering": float(
                clustering_stats(train_graphs, real_graphs, compute_emd=False)
            ),
            "spectral": float(
                spectral_stats(train_graphs, real_graphs, compute_emd=False)
            ),
        }
        setattr(args, "_checkpoint_proxy_reference", reference_metrics)
    ratios = []
    for key in ("degree", "clustering", "spectral"):
        denominator = round(float(reference_metrics[key]), 4)
        if denominator != 0.0:
            ratios.append(generated_metrics[key] / denominator)
    proxy_ratio = float(np.mean(ratios)) if ratios else float("inf")
    result = dict(generated_metrics)
    result.update(vun_metrics)
    result["proxy_ratio"] = proxy_ratio
    return result


@torch.no_grad()
def vae_validation_metrics(
    args, model, val_graphs, val_adj, profile, device, train_core_adj=None
):
    """Evaluate teacher-forced reconstructions on validation graphs.
    
    Planar and Tree rank by V.U.N. followed by a structural-ratio proxy;
    Grid uses average MMD. Validation features and padding dimensions must
    match the training dataset. Missing validation data produces infinite MMD."""
    model.eval()
    if val_graphs is None or len(val_graphs.list_adjs) == 0:
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}

    cap = int(getattr(args, "val_eval_subset", 0) or 0)
    n_total = len(val_graphs.list_adjs)
    n_eval = n_total if cap <= 0 else min(cap, n_total)

    real_graphs = []
    for adj in val_adj[:n_eval]:
        G = _adj_to_nx(adj)
        if G.number_of_nodes() > 0:
            real_graphs.append(G)
    if not real_graphs:
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}

    # Reuse the dataset's existing feature pipeline.
    val_graphs.processALL(self_for_none=True)
    # Datasets.__init__ leaves featureList=None until set_features() is called.
    # get__(..., bfs=None) iterates over self.featureList, so we must set it to
    # an empty list (validation does not use kernel features anyway).
    if getattr(val_graphs, "featureList", None) is None:
        val_graphs.set_features([])
    pred_graphs_all = []
    mini_bs = max(1, int(getattr(args, "batchSize", 64)))
    for from_ in range(0, n_eval, mini_bs):
        to_ = min(from_ + mini_bs, n_eval)
        org_adj, x_s, node_num_list, _, _, _ = val_graphs.get__(
            from_, to_, True, bfs=None
        )
        if len(org_adj) == 0:
            continue
        x_s = torch.cat(x_s).reshape(-1, x_s[0].shape[-1]).to(device)
        batch_size = [len(org_adj), org_adj[0].shape[0]]
        node_mask = build_node_mask(node_num_list, batch_size[1], device)
        _, subgraphs = get_subGraph_features(args, org_adj, None, None)
        subgraphs = subgraphs.to(device)
        org_adj_dgl = prepare_batch_graphs(org_adj, device)

        profile_vec = None
        if profile is not None and getattr(model.encoder, "use_struct_cond", False):
            raw_scalar = compute_profile_scalar_stats(subgraphs, node_mask)
            std_scalar = profile.standardize_scalar(raw_scalar)
            hist = _batch_degree_histogram(subgraphs, node_mask, profile.n_bins)
            profile_vec = torch.cat([std_scalar, hist], dim=-1)

        _, _, _, _, _, aux_preds, edge_logits = model(
            org_adj_dgl, x_s, batch_size, node_mask=node_mask, profile_vec=profile_vec,
        )
        # Teacher-forced (n, E) from ground truth so decoding aims correctly.
        stats_target = compute_graph_batch_stats(subgraphs, node_mask)

        preds = decode_samples_to_graphs(
            args, torch.sigmoid(edge_logits), aux_preds["node_logits"], stats_target,
            degree_pred=aux_preds.get("degree_pred"),
            edge_logits_raw=edge_logits,
            profile=profile,
            sampled_profile_hist=(profile_vec[:, 8:] if profile_vec is not None else None),
        )
        pred_graphs_all.extend([g for g in preds if g.number_of_nodes() > 0])

    if not pred_graphs_all:
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}

    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    if structure_constraint in {"planar", "tree"}:
        train_graphs = [
            _adj_to_nx(adjacency) for adjacency in (train_core_adj or [])
        ]
        try:
            return benchmark_validation_metrics(
                args, real_graphs, pred_graphs_all, train_graphs
            )
        except Exception as exc:
            print(f"[val] benchmark proxy computation failed: {exc!r}")
            return {
                "vun": 0.0,
                "degree": float("inf"),
                "clustering": float("inf"),
                "spectral": float("inf"),
                "proxy_ratio": float("inf"),
            }

    try:
        deg = degree_stats(real_graphs, pred_graphs_all, compute_emd=True)
        clu = clustering_stats(real_graphs, pred_graphs_all, compute_emd=True)
        spc = spectral_stats(real_graphs, pred_graphs_all, compute_emd=True)
    except Exception as exc:
        print(f"[val] MMD computation failed: {exc!r}")
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}
    avg = (deg + clu + spc) / 3.0
    return {"degree": float(deg), "clustering": float(clu),
            "spectral": float(spc), "avg_mmd": float(avg)}


@torch.no_grad()
def flow_validation_metrics(
    args,
    flow_model,
    decoder,
    cond_bank,
    val_adj,
    profile,
    device,
    condition_profile_bank=None,
    train_core_adj=None,
):
    """Generate a validation-sized sample and compare it with validation graphs.
    
    Candidate reranking is reserved for final evaluation to limit validation cost."""
    cap = int(getattr(args, "val_eval_subset", 0) or 0)
    val_adj_use = _val_subset(val_adj, cap)
    real_graphs = [_adj_to_nx(a) for a in val_adj_use if a is not None]
    real_graphs = [g for g in real_graphs if g.number_of_nodes() > 0]
    if not real_graphs:
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}

    n_samples = len(val_adj_use)
    pred_graphs, _, _ = sample_and_decode(
        args,
        flow_model,
        decoder,
        cond_bank,
        n_samples,
        device,
        profile=profile,
        condition_profile_bank=condition_profile_bank,
    )
    pred_graphs = [g for g in pred_graphs if g.number_of_nodes() > 0]
    if not pred_graphs:
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}

    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    if structure_constraint in {"planar", "tree"}:
        train_graphs = [
            _adj_to_nx(adjacency) for adjacency in (train_core_adj or [])
        ]
        try:
            return benchmark_validation_metrics(
                args, real_graphs, pred_graphs, train_graphs
            )
        except Exception as exc:
            print(f"[flow-val] benchmark proxy computation failed: {exc!r}")
            return {
                "vun": 0.0,
                "degree": float("inf"),
                "clustering": float("inf"),
                "spectral": float("inf"),
                "proxy_ratio": float("inf"),
            }

    try:
        deg = degree_stats(real_graphs, pred_graphs, compute_emd=True)
        clu = clustering_stats(real_graphs, pred_graphs, compute_emd=True)
        spc = spectral_stats(real_graphs, pred_graphs, compute_emd=True)
    except Exception as exc:
        print(f"[flow-val] MMD computation failed: {exc!r}")
        return {"degree": float("inf"), "clustering": float("inf"),
                "spectral": float("inf"), "avg_mmd": float("inf")}
    avg = (deg + clu + spc) / 3.0
    return {"degree": float(deg), "clustering": float(clu),
            "spectral": float(spc), "avg_mmd": float(avg)}


def rerank_candidates_by_profile(
    graphs, profile, target_n: int, prefer_unique: bool = False
):
    """Keep top-N profile candidates, optionally preferring unique outputs.

    Uniqueness is checked only among generated candidates.  Training graphs are
    deliberately not consulted here; novelty remains exclusively an evaluation
    metric, as in SimGFM.
    """
    if not graphs:
        return graphs
    if target_n >= len(graphs) and not prefer_unique:
        return list(graphs)
    scored = [(profile.score_graph(g), g) for g in graphs]
    scored.sort(key=lambda x: x[0])
    if not prefer_unique:
        return [g for _, g in scored[:target_n]]

    selected = []
    deferred_duplicates = []
    for _, graph in scored:
        duplicate = any(
            nx.faster_could_be_isomorphic(graph, old)
            and nx.is_isomorphic(graph, old)
            for old in selected
        )
        if duplicate:
            deferred_duplicates.append(graph)
            continue
        selected.append(graph)
        if len(selected) == target_n:
            return selected

    # If the candidate pool contains fewer than target_n isomorphism classes,
    # fill the remainder in profile-score order rather than returning too few.
    selected.extend(deferred_duplicates[: max(0, target_n - len(selected))])
    return selected[:target_n]


def train_klein_encoder(args, list_graphs, val_adj, train_core_adj, val_graphs, device):
    """Train the graph autoencoder with structural conditioning.
    
    Structural profiles are fitted on training graphs only. The objective
    combines reconstruction and auxiliary degree losses. Validation scores
    determine checkpoint selection when validation is enabled."""
    print("\n" + "=" * 60)
    print("Phase 1: Training Klein Encoder")
    print("=" * 60)

    graph_save_path = args.graph_save_path
    epoch_number = args.epoch_number
    mini_batch_size = args.batchSize

    in_feature_dim = list_graphs.feature_size
    node_num = list_graphs.max_num_nodes
    graph_em_dim = args.graphEmDim
    cond_dim = getattr(args, "cond_dim", 64)
    decoder_node_dim = getattr(args, "decoder_node_dim", 128)
    encoder_blocks = getattr(args, "encoder_blocks", 4)

    use_struct_cond = bool(getattr(args, "use_struct_cond", True))
    struct_cond_dim = int(getattr(args, "struct_cond_dim", 32))

    # ---- Fit DatasetProfile on train-core ----
    profile = DatasetProfile(n_bins=32)
    profile.fit(train_core_adj)
    torch.save(profile.state_dict(), graph_save_path + "dataset_profile.pt")
    profile_stat_dim = profile.profile_stat_dim  # 8 scalars + 32 bins = 40
    print(f"[profile] scalar_mean={profile.scalar_mean.tolist()[:4]}... "
          f"n_min={profile.n_min} n_max={profile.n_max} m_min={profile.m_min} m_max={profile.m_max}")

    hidden_dim = max(128, min(256, graph_em_dim * 4))
    encoder = KleinEncoder(
        in_feature_dim=in_feature_dim,
        hidden_layers=[hidden_dim] * encoder_blocks,
        graph_latent_dim=graph_em_dim,
        dropout=args.dropout,
        cond_dim=cond_dim,
        encoder_blocks=encoder_blocks,
        input_proj_dim=64,
        use_struct_cond=use_struct_cond,
        profile_stat_dim=profile_stat_dim,
        struct_cond_dim=struct_cond_dim,
    )

    full_cond_dim = encoder.full_cond_dim
    decoder = MaskedGraphDecoder(
        latent_dim=graph_em_dim,
        cond_dim=full_cond_dim,
        max_nodes=node_num,
        directed=args.directed,
        node_dim=decoder_node_dim,
    )

    degree_center = torch.tensor([[x] for x in range(0, node_num, 1)])
    degree_width = torch.tensor([[0.1] for _ in range(0, node_num, 1)])
    bin_center = torch.tensor([[x] for x in range(0, node_num, 1)])
    bin_width = torch.tensor([[1] for _ in range(0, node_num, 1)])

    kernel_model = kernel(
        device=device, kernel_type=[], step_num=0,
        bin_width=bin_width, bin_center=bin_center,
        degree_bin_center=degree_center, degree_bin_width=degree_width
    )

    model = KleinGraphVAE(
        encoder=encoder,
        decoder=decoder,
        kernel_model=kernel_model,
        auto_encoder=False
    )
    model.to(device)

    torch.save(model, graph_save_path + 'klein_model.pt')
    torch.save(kernel_model, graph_save_path + 'kernel.pt')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_number)

    self_for_none = True
    list_graphs.processALL(self_for_none=self_for_none)
    adj_list = list_graphs.get_adj_list()
    graph_features, _ = get_subGraph_features(args, adj_list, None, kernel_model)
    list_graphs.set_features(graph_features)

    degree_reg_weight = float(getattr(args, "degree_reg_weight", 0.10))
    degree_aux_weight = float(getattr(args, "degree_aux_weight", 0.05))
    degree_hist_budget_calibration = bool(
        getattr(args, "degree_hist_budget_calibration", False)
    )
    degree_hist_temperature = float(getattr(args, "degree_hist_temperature", 0.5))
    val_eval_interval = int(getattr(args, "val_eval_interval", 200))

    # Dataset-aware bin scale for the soft degree-histogram aux loss.
    # Profile.scalar_mean/std fields are ordered:
    #   [n, m, density, avg_deg, deg_mom2, deg_mom3, deg_mom4, max_deg]
    # so index 7 is the max-degree statistic.
    mean_max_deg = float(profile.scalar_mean[7].item())
    std_max_deg = float(profile.scalar_std[7].item())
    degree_hist_max = max(1.0, mean_max_deg + 2.0 * std_max_deg)
    print(f"[profile] degree_hist_max = {degree_hist_max:.2f}  "
          f"(mean_max_deg={mean_max_deg:.2f}, std={std_max_deg:.2f})")

    min_loss = float('inf')
    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    best_val_key = None
    best_embeddings = None
    best_cond_codes = None

    for epoch in range(epoch_number):
        list_graphs.shuffle()
        epoch_losses = []

        for batch_idx in range(0, len(list_graphs.list_adjs), mini_batch_size):
            from_ = batch_idx
            to_ = min(from_ + mini_batch_size, len(list_graphs.list_adjs))

            org_adj, x_s, node_num_list, _, target_kernel_val, graph_stats = list_graphs.get__(
                from_, to_, self_for_none, bfs=None
            )
            if len(org_adj) == 0:
                continue

            x_s = torch.cat(x_s).reshape(-1, x_s[0].shape[-1]).to(device)
            batch_size = [len(org_adj), org_adj[0].shape[0]]
            node_mask = build_node_mask(node_num_list, batch_size[1], device)
            _, subgraphs = get_subGraph_features(args, org_adj, None, None)
            subgraphs = subgraphs.to(device)
            stats_target = torch.stack(graph_stats).to(device)

            # Structural conditioning vector (standardized scalars + raw hist).
            profile_vec = None
            if use_struct_cond:
                raw_scalar = compute_profile_scalar_stats(subgraphs, node_mask)
                std_scalar = profile.standardize_scalar(raw_scalar)
                hist = _batch_degree_histogram(subgraphs, node_mask, profile.n_bins)
                profile_vec = torch.cat([std_scalar, hist], dim=-1)

            org_adj_dgl = prepare_batch_graphs(org_adj, device)

            model.train()
            optimizer.zero_grad()

            reconstructed_adj, samples, mean, log_std, cond_code, aux_preds, adj_logits = model(
                org_adj_dgl, x_s, batch_size, node_mask=node_mask, profile_vec=profile_vec,
            )

            stats_target = compute_graph_batch_stats(subgraphs, node_mask)
            losses = compute_vae_loss(
                adj_logits=adj_logits,
                adj_probs=reconstructed_adj,
                target_adj=subgraphs,
                node_logits=aux_preds["node_logits"],
                node_mask=node_mask,
                stats_pred=aux_preds["stats_pred"],
                stats_target=stats_target,
                encoder_aux=aux_preds["encoder_aux"],
                log_std=log_std,
                mean=mean,
                kernel_model=kernel_model,
                target_kernel_val=[val.to(device) for val in target_kernel_val],
                degree_pred=aux_preds.get("degree_pred"),
                degree_reg_weight=degree_reg_weight,
                degree_aux_weight=degree_aux_weight,
                degree_hist_max=degree_hist_max,
                degree_hist_budget_calibration=degree_hist_budget_calibration,
                degree_hist_temperature=degree_hist_temperature,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(losses)

            total_loss_item = losses["total"].item()
            if total_loss_item < min_loss:
                min_loss = total_loss_item
                torch.save(model.state_dict(), graph_save_path + "klein_model_lowloss.pt")

        scheduler.step()

        # Periodic deterministic encoder dump (kept for backward compatibility).
        if (epoch + 1) % max(1, val_eval_interval) == 0 or epoch == epoch_number - 1:
            best_embeddings, best_cond_codes = collect_encoder_outputs(
                args, model, list_graphs, device, profile=profile,
            )
            torch.save(best_embeddings, graph_save_path + f'{epoch}_klein_feat.pt')
            torch.save(best_cond_codes, graph_save_path + f'{epoch}_cond_feat.pt')

            # validation-based checkpoint selection.
            if val_graphs is not None and len(val_adj) > 0:
                val_metrics = vae_validation_metrics(
                    args,
                    model,
                    val_graphs,
                    val_adj,
                    profile,
                    device,
                    train_core_adj=train_core_adj,
                )
                candidate_key = checkpoint_selection_key(
                    val_metrics, structure_constraint
                )
                if structure_constraint in {"planar", "tree"}:
                    print(
                        f"[val] epoch={epoch+1}  vun={val_metrics.get('vun', 0.0):.4f}  "
                        f"proxy_ratio={val_metrics.get('proxy_ratio', float('inf')):.4f}  "
                        f"deg={val_metrics['degree']:.4f}  "
                        f"clu={val_metrics['clustering']:.4f}  "
                        f"spec={val_metrics['spectral']:.4f}"
                    )
                else:
                    avg_val = val_metrics["avg_mmd"]
                    print(
                        f"[val] epoch={epoch+1}  deg={val_metrics['degree']:.4f}  "
                        f"clu={val_metrics['clustering']:.4f}  "
                        f"spec={val_metrics['spectral']:.4f}  avg={avg_val:.4f}"
                    )
                if best_val_key is None or candidate_key < best_val_key:
                    best_val_key = candidate_key
                    torch.save(model.state_dict(), graph_save_path + "klein_encoder_best.pt")
                    print(
                        f"[val]   new best checkpoint key={candidate_key}; "
                        "saved klein_encoder_best.pt"
                    )

        if epoch_losses:
            edge = np.mean([loss["edge"].item() for loss in epoch_losses])
            node = np.mean([loss["node"].item() for loss in epoch_losses])
            stats = np.mean([loss["stats"].item() for loss in epoch_losses])
            kl = np.mean([loss["kl"].item() for loss in epoch_losses])
            acc = np.mean([loss["acc"].item() for loss in epoch_losses])
            total = np.mean([loss["total"].item() for loss in epoch_losses])
            dreg = np.mean([loss.get("degree_reg", torch.tensor(0.0)).item() for loss in epoch_losses])
            dhist = np.mean([loss.get("degree_hist", torch.tensor(0.0)).item() for loss in epoch_losses])
            print(
                f"Epoch {epoch + 1}/{epoch_number}, Total: {total:.6f}, Edge: {edge:.6f}, "
                f"Node: {node:.6f}, Stats: {stats:.6f}, KL: {kl:.6f}, "
                f"DegReg: {dreg:.4f}, DegHist: {dhist:.4f}, Acc: {acc:.4f}"
            )

    # Load best validation checkpoint if available, else fall back to last.
    best_model_path = graph_save_path + "klein_encoder_best.pt"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best-val encoder checkpoint from {best_model_path}")
    else:
        print("[val] no validation checkpoint saved; keeping last-epoch weights.")

    # Final embeddings (train-core only) for flow training.
    best_embeddings, best_cond_codes = collect_encoder_outputs(
        args, model, list_graphs, device, profile=profile,
    )
    torch.save(best_embeddings, graph_save_path + "klein_feat_best.pt")
    torch.save(best_cond_codes, graph_save_path + "cond_feat_best.pt")

    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    aligned_profile_bank = (
        profile.aligned_bank(list_graphs.list_adjs)
        if structure_constraint in {"planar", "tree"}
        else None
    )

    torch.save(model.state_dict(), graph_save_path + "klein_encoder_final.pt")
    torch.save(model, graph_save_path + "klein_model_trained.pt")

    return model, best_embeddings, best_cond_codes, profile, aligned_profile_bank


def train_klein_flow_matching(
    args, klein_embeddings, cond_codes, device,
    encoder_model=None, val_adj=None, profile=None, condition_profile_bank=None,
    train_core_adj=None,
):
    """Fit conditional Flow Matching to training-set graph encodings.
    
    Latent normalization statistics are estimated from training embeddings.
    The encoder, validation graphs, and structural profile support optional
    validation-based selection of the Flow Matching checkpoint."""
    print("\n" + "=" * 60)
    print("Phase 2: Training Klein Flow Matching")
    print("=" * 60)

    graph_save_path = args.graph_save_path
    dim = klein_embeddings.shape[1]
    cond_dim = cond_codes.shape[1]

    flow_model = KleinFlowMatching(
        dim=dim,
        cond_dim=cond_dim,
        hidden_dim=getattr(args, 'dit_hidden_dim', 512),
        num_heads=getattr(args, 'dit_num_heads', 8),
        num_layers=getattr(args, 'dit_num_layers', 6),
        num_timesteps=getattr(args, 'flow_steps', 200),
        base_std=getattr(args, 'flow_base_std', 1.0),
        use_simple_dit=getattr(args, 'use_simple_dit', False),
        flow_cond_dropout=getattr(args, 'flow_cond_dropout', 0.1),
        use_cond_guidance=getattr(args, 'use_cond_guidance', True),
        guidance_scale=getattr(args, 'flow_guidance_scale', 1.2),
        device=device
    )
    flow_model.to(device)

    # set_latent_stats on train-core encodings only.
    tangent = flow_model.to_tangent_space(klein_embeddings.to(device))
    flow_model.set_latent_stats(
        tangent.mean(dim=0),
        tangent.std(dim=0).clamp_min(1e-4)
    )

    epochs = getattr(args, 'epoch_diff', 1000)
    batch_size = getattr(args, 'batchSize', 64)
    lr = getattr(args, 'lr_diff', 1e-4)
    flow_val_eval_interval = int(getattr(args, "flow_val_eval_interval", 400))

    optimizer = optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    klein_embeddings = klein_embeddings.to(device)
    cond_codes = cond_codes.to(device)
    num_samples = klein_embeddings.shape[0]

    print(f"Training Klein Flow Matching on {num_samples} samples, dim={dim}, cond_dim={cond_dim}")

    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    best_val_key = None

    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        if condition_profile_bank is not None:
            # Keep the canonical condition-code order aligned with the profile
            # bank used by constrained validation.  Only the epoch-local views
            # are permuted for minibatch training.
            epoch_embeddings = klein_embeddings[perm]
            epoch_cond_codes = cond_codes[perm]
        else:
            # Grid conditions are perturbed in place within the sampled batch.
            klein_embeddings = klein_embeddings[perm]
            cond_codes = cond_codes[perm]
            epoch_embeddings = klein_embeddings
            epoch_cond_codes = cond_codes

        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch = epoch_embeddings[i:i + batch_size]
            cond_batch = epoch_cond_codes[i:i + batch_size]

            optimizer.zero_grad()
            loss = flow_model.loss_fn(batch, cond_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # flow-stage validation checkpointing.
        if (
            encoder_model is not None and val_adj is not None and len(val_adj) > 0
            and (epoch + 1) % max(1, flow_val_eval_interval) == 0
        ):
            decoder = encoder_model.decoder.to(device)
            decoder.eval()
            val_metrics = flow_validation_metrics(
                args, flow_model, decoder, cond_codes, val_adj, profile, device,
                condition_profile_bank=condition_profile_bank,
                train_core_adj=train_core_adj,
            )
            candidate_key = checkpoint_selection_key(
                val_metrics, structure_constraint
            )
            if structure_constraint in {"planar", "tree"}:
                print(
                    f"[flow-val] epoch={epoch+1}  "
                    f"vun={val_metrics.get('vun', 0.0):.4f}  "
                    f"proxy_ratio={val_metrics.get('proxy_ratio', float('inf')):.4f}  "
                    f"deg={val_metrics['degree']:.4f}  "
                    f"clu={val_metrics['clustering']:.4f}  "
                    f"spec={val_metrics['spectral']:.4f}"
                )
            else:
                avg_val = val_metrics["avg_mmd"]
                print(
                    f"[flow-val] epoch={epoch+1}  deg={val_metrics['degree']:.4f}  "
                    f"clu={val_metrics['clustering']:.4f}  "
                    f"spec={val_metrics['spectral']:.4f}  avg={avg_val:.4f}"
                )
            if best_val_key is None or candidate_key < best_val_key:
                best_val_key = candidate_key
                torch.save(flow_model.state_dict(), graph_save_path + "klein_flow_best.pt")
                print(
                    f"[flow-val]   new best checkpoint key={candidate_key}; "
                    "saved klein_flow_best.pt"
                )

    torch.save(flow_model, graph_save_path + 'klein_flow_model.pt')
    torch.save(flow_model.state_dict(), graph_save_path + 'klein_flow_model_state.pt')
    torch.save(flow_model.state_dict(), graph_save_path + 'klein_flow_last.pt')

    # Reload best validation checkpoint if it exists.
    best_path = graph_save_path + "klein_flow_best.pt"
    if os.path.exists(best_path):
        flow_model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[flow-val] loaded best validation checkpoint from {best_path}")

    return flow_model


def _decode_one_threshold(adj_probs_i, node_probs_i, stats_pred_i, directed, max_nodes):
    """Decode edges at probability 0.5 for threshold-based ablations."""
    predicted_nodes = int(torch.clamp(torch.round(stats_pred_i[0]), 1, max_nodes).item())
    confident_nodes = int(node_probs_i.gt(0.5).sum().item())
    node_count = max(1, min(max_nodes, confident_nodes if confident_nodes > 0 else predicted_nodes))

    topk = torch.topk(node_probs_i, k=node_count).indices
    sub_adj = adj_probs_i[topk][:, topk]
    sub_adj.fill_diagonal_(0.0)

    binary_adj = (sub_adj > 0.5).float().numpy()
    if not directed:
        binary_adj = np.triu(binary_adj, 1)
        binary_adj = binary_adj + binary_adj.T
    return binary_adj, topk


def _decode_one_topE(
    adj_probs_i, node_probs_i, stats_pred_i, degree_pred_i,
    directed, max_nodes, n_min, n_max,
    rel_tol, abs_tol,
):
    """Top-E edge-budget decoding.

    1. n = round(stats_pred_i[0]) clamped to [n_min, n_max].
    2. Pick top-n node slots by node_probs.
    3. E = round(stats_pred_i[1]) clamped to [0, n*(n-1)/2].
    4. degree-aware candidate mask (top d_i incident per node, OR-symmetrized).
    5. trim/add to hit E if within tolerance; else fall back to global top-E.
    """
    # 1. Node count.
    if n_max is None or n_max <= 0:
        n_max = max_nodes
    n_min = max(1, int(n_min) if n_min is not None else 1)
    n_max = max(n_min, min(max_nodes, int(n_max)))
    n_pred = int(torch.clamp(torch.round(stats_pred_i[0]), n_min, n_max).item())

    # 2. Top-n slots by node_probs.
    topk = torch.topk(node_probs_i, k=n_pred).indices
    sub_adj = adj_probs_i[topk][:, topk].clone()
    sub_adj.fill_diagonal_(0.0)

    n = n_pred
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32), topk

    # 3. Target edge count.
    max_E = n * (n - 1) // 2
    E_target = int(torch.clamp(torch.round(stats_pred_i[1]), 0, max_E).item())

    # Symmetrize once for undirected datasets so logits are comparable.
    if not directed:
        sub_adj = 0.5 * (sub_adj + sub_adj.t())
    sub_np = sub_adj.numpy()
    iu = np.triu_indices(n, k=1)
    upper_logits = sub_np[iu]   # length max_E

    if E_target <= 0:
        return np.zeros((n, n), dtype=np.float32), topk

    # 4. Degree-aware candidate mask via per-node top-d_i.
    candidate_mask = np.zeros((n, n), dtype=bool)
    if degree_pred_i is not None and degree_pred_i.numel() >= n:
        deg_pred = degree_pred_i[topk].numpy()
        for i in range(n):
            d_i = int(np.clip(np.round(deg_pred[i]), 0, n - 1))
            if d_i == 0:
                continue
            # neighbours scored by sub_np[i, :]
            row = sub_np[i].copy()
            row[i] = -np.inf
            nbrs = np.argpartition(-row, kth=d_i - 1)[:d_i]
            candidate_mask[i, nbrs] = True
        # OR-symmetrize.
        candidate_mask = candidate_mask | candidate_mask.T
        np.fill_diagonal(candidate_mask, False)

    # Convert to upper-tri candidate set.
    cand_upper = candidate_mask[iu]
    E_cand = int(cand_upper.sum())

    tol = max(int(abs_tol), int(round(rel_tol * E_target)))

    chosen = np.zeros_like(upper_logits, dtype=bool)
    if E_cand > 0 and abs(E_cand - E_target) <= tol:
        chosen = cand_upper.copy()
        if E_cand > E_target:
            # drop lowest-logit candidates
            cand_idx = np.where(cand_upper)[0]
            drop_n = E_cand - E_target
            drop = cand_idx[np.argsort(upper_logits[cand_idx])[:drop_n]]
            chosen[drop] = False
        elif E_cand < E_target:
            # add highest-logit non-candidates
            non_cand_idx = np.where(~cand_upper)[0]
            add_n = E_target - E_cand
            if non_cand_idx.size > 0:
                add = non_cand_idx[np.argsort(-upper_logits[non_cand_idx])[:add_n]]
                chosen[add] = True
    else:
        # Global top-E fallback.
        topE = np.argpartition(-upper_logits, kth=min(E_target, upper_logits.size - 1))[:E_target]
        chosen[topE] = True

    # Build binary adjacency on n nodes.
    binary_adj = np.zeros((n, n), dtype=np.float32)
    binary_adj[iu[0][chosen], iu[1][chosen]] = 1.0
    binary_adj = binary_adj + binary_adj.T
    np.fill_diagonal(binary_adj, 0.0)
    return binary_adj, topk


def _connectivity_repair(
    G: nx.Graph,
    edge_logits: np.ndarray,
    target_E: int,
    rel_tol: float,
    abs_tol: int,
) -> nx.Graph:
    """Bridge-aware connectivity repair.

    Args:
        G: candidate graph (already on top-n nodes labelled [0..n-1]).
        edge_logits: dense (n, n) symmetric logit matrix.
        target_E: desired edge count.
        rel_tol/abs_tol: trim tolerance.

    Returns:
        Repaired graph. Keeps all nodes; removes self-loops; tries to connect
        components by adding highest-logit cut edges; trims with non-bridge
        edges to honor budget; terminates when only bridges remain.
    """
    if G.number_of_nodes() <= 1:
        return G
    G.remove_edges_from(nx.selfloop_edges(G))

    components = list(nx.connected_components(G))
    C = len(components)
    if C <= 1:
        # already connected; may still trim below budget
        max_repair = 0
    else:
        max_repair = max(C - 1, int(np.ceil(rel_tol * target_E)))

    repaired = 0
    while C > 1 and repaired < max_repair:
        components.sort(key=len, reverse=True)
        A, B = components[0], components[1]
        best_logit = -np.inf
        best_pair = None
        for u in A:
            for v in B:
                if u == v or G.has_edge(u, v):
                    continue
                if edge_logits[u, v] > best_logit:
                    best_logit = edge_logits[u, v]
                    best_pair = (u, v)
        if best_pair is None:
            break
        G.add_edge(*best_pair)
        repaired += 1
        components = list(nx.connected_components(G))
        C = len(components)

    # Trim if over budget.
    excess = G.number_of_edges() - target_E
    excess_tol = max(int(abs_tol), int(round(rel_tol * target_E)))
    if excess > excess_tol:
        removals = 0
        bridges = set(map(tuple, map(sorted, nx.bridges(G)))) if G.number_of_edges() else set()
        edges = [(u, v) for u, v in G.edges()]
        non_bridges = [e for e in edges if tuple(sorted(e)) not in bridges]
        # Sort non-bridges by ascending logit.
        non_bridges.sort(key=lambda e: edge_logits[e[0], e[1]])
        while excess > excess_tol and non_bridges:
            u, v = non_bridges.pop(0)
            if G.has_edge(u, v):
                G.remove_edge(u, v)
                excess -= 1
                removals += 1
                if removals % 5 == 0:
                    # Re-amortize bridge recompute every 5 removals.
                    bridges = set(map(tuple, map(sorted, nx.bridges(G)))) if G.number_of_edges() else set()
                    non_bridges = [e for e in G.edges() if tuple(sorted(e)) not in bridges]
                    non_bridges.sort(key=lambda e: edge_logits[e[0], e[1]])
    return G


def project_degree_budget(
    degree_values,
    total_degree: int,
    min_degree: int = 0,
    max_degree=None,
) -> np.ndarray:
    """Project real-valued degree targets to an exact integer budget.

    The projection preserves the relative shape of the predictions by first
    finding a shared clipped offset and then allocating the remaining integer
    units by fractional residual.  Constrained decoders use ``min_degree=1``
    because both trees and connected planar graphs have no isolated vertices.
    """
    values = np.asarray(degree_values, dtype=np.float64).reshape(-1)
    node_count = int(values.size)
    if node_count == 0:
        if int(total_degree) != 0:
            raise ValueError("A non-zero degree budget needs at least one node")
        return np.zeros(0, dtype=np.int64)
    if max_degree is None:
        max_degree = max(node_count - 1, 0)
    min_degree = int(min_degree)
    max_degree = int(max_degree)
    total_degree = int(total_degree)
    minimum_sum = node_count * min_degree
    maximum_sum = node_count * max_degree
    if min_degree < 0 or max_degree < min_degree:
        raise ValueError(
            f"Invalid degree bounds: min={min_degree}, max={max_degree}"
        )
    if not minimum_sum <= total_degree <= maximum_sum:
        raise ValueError(
            "Degree budget is infeasible: "
            f"total={total_degree}, feasible=[{minimum_sum}, {maximum_sum}]"
        )

    midpoint = 0.5 * (min_degree + max_degree)
    values = np.nan_to_num(
        values, nan=midpoint, posinf=max_degree, neginf=min_degree
    )
    values = np.clip(values, min_degree, max_degree)

    # Find lambda such that sum(clip(values + lambda)) == total_degree in the
    # continuous relaxation.  Wide finite bounds cover every clipped solution.
    low = float(min_degree - np.max(values) - max_degree - 1.0)
    high = float(max_degree - np.min(values) + max_degree + 1.0)
    for _ in range(80):
        offset = 0.5 * (low + high)
        projected = np.clip(values + offset, min_degree, max_degree)
        if float(projected.sum()) < total_degree:
            low = offset
        else:
            high = offset
    continuous = np.clip(values + 0.5 * (low + high), min_degree, max_degree)
    integer = np.floor(continuous + 1e-10).astype(np.int64)
    remainder = total_degree - int(integer.sum())
    if remainder > 0:
        fractional = continuous - integer
        candidates = np.where(integer < max_degree)[0]
        order = candidates[np.lexsort((candidates, -fractional[candidates]))]
        if remainder > order.size:
            raise RuntimeError("Degree projection could not allocate its remainder")
        integer[order[:remainder]] += 1
    elif remainder < 0:
        fractional = continuous - integer
        candidates = np.where(integer > min_degree)[0]
        order = candidates[np.lexsort((candidates, fractional[candidates]))]
        if -remainder > order.size:
            raise RuntimeError("Degree projection could not remove its excess")
        integer[order[:-remainder]] -= 1

    if int(integer.sum()) != total_degree:
        raise RuntimeError(
            f"Degree projection missed budget {total_degree}: got {integer.sum()}"
        )
    return integer


def degree_histogram_quantiles(histogram, node_count: int) -> np.ndarray:
    """Expand a normalized degree histogram into ``node_count`` quantiles."""
    histogram = np.asarray(histogram, dtype=np.float64).reshape(-1)
    node_count = int(node_count)
    if node_count <= 0:
        return np.zeros(0, dtype=np.float64)
    if histogram.size == 0:
        return np.zeros(node_count, dtype=np.float64)
    histogram = np.nan_to_num(histogram, nan=0.0, posinf=0.0, neginf=0.0)
    histogram = np.clip(histogram, 0.0, None)
    if float(histogram.sum()) <= 0:
        histogram = np.ones_like(histogram)
    histogram = histogram / histogram.sum()
    cumulative = np.cumsum(histogram)
    quantiles = (np.arange(node_count, dtype=np.float64) + 0.5) / node_count
    bin_indices = np.searchsorted(cumulative, quantiles, side="left")
    bin_indices = np.clip(bin_indices, 0, histogram.size - 1)
    # DatasetProfile histograms partition [0, n-1] into equally sized bins.
    return (bin_indices + 0.5) * max(node_count - 1, 0) / histogram.size


def blend_degree_targets(
    predicted_degrees,
    profile_histogram,
    blend: float,
    total_degree: int,
    min_degree: int = 1,
) -> np.ndarray:
    """Rank-align model degrees with a profile histogram and project to 2E."""
    predicted = np.asarray(predicted_degrees, dtype=np.float64).reshape(-1)
    node_count = int(predicted.size)
    if node_count == 0:
        return project_degree_budget([], total_degree, min_degree, 0)
    blend = float(blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"degree_profile_blend must be in [0, 1], got {blend}")
    predicted = np.nan_to_num(
        predicted, nan=0.0, posinf=node_count - 1, neginf=0.0
    )
    predicted = np.clip(predicted, 0.0, max(node_count - 1, 0))
    profile_sorted = np.sort(degree_histogram_quantiles(profile_histogram, node_count))
    rank_order = np.argsort(predicted, kind="mergesort")
    profile_by_node = np.empty(node_count, dtype=np.float64)
    profile_by_node[rank_order] = profile_sorted
    blended = (1.0 - blend) * predicted + blend * profile_by_node
    return project_degree_budget(
        blended,
        total_degree=total_degree,
        min_degree=min_degree,
        max_degree=max(node_count - 1, min_degree),
    )


def _noisy_symmetric_scores(edge_scores, noise_scale: float) -> np.ndarray:
    scores = np.asarray(edge_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Expected square edge scores, got {scores.shape}")
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, -np.inf)
    noise_scale = float(noise_scale)
    if noise_scale < 0:
        raise ValueError(f"constraint_noise_scale must be non-negative, got {noise_scale}")
    if noise_scale > 0:
        uniform = np.random.random_sample(scores.shape)
        uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
        gumbel = -np.log(-np.log(uniform))
        gumbel = 0.5 * (gumbel + gumbel.T)
        scores = scores + noise_scale * gumbel
        np.fill_diagonal(scores, -np.inf)
    return scores


def decode_tree_constrained(
    edge_scores,
    target_degrees,
    noise_scale: float = 0.0,
) -> nx.Graph:
    """Build a weighted Prüfer tree with the requested exact degree sequence."""
    target = np.asarray(target_degrees, dtype=np.int64).reshape(-1)
    node_count = int(target.size)
    if node_count < 2:
        raise ValueError("Tree constrained decoding requires at least two nodes")
    if np.any(target < 1) or np.any(target > node_count - 1):
        raise ValueError(f"Invalid tree degree sequence: {target.tolist()}")
    if int(target.sum()) != 2 * (node_count - 1):
        raise ValueError(
            "Tree degree sequence must sum to 2(n-1): "
            f"sum={target.sum()}, n={node_count}"
        )
    scores = _noisy_symmetric_scores(edge_scores, noise_scale)
    if scores.shape[0] != node_count:
        raise ValueError(
            f"Tree scores/degree size mismatch: {scores.shape[0]} vs {node_count}"
        )

    # A tree degree sequence maps to a Prüfer multiset in which node i occurs
    # target_degree[i]-1 times.  We choose the order of that multiset jointly
    # with the current leaf, maximizing the decoder's edge score each step.
    occurrences = target - 1
    active = np.ones(node_count, dtype=bool)
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for _ in range(node_count - 2):
        leaves = np.flatnonzero(active & (occurrences == 0))
        codes = np.flatnonzero(active & (occurrences > 0))
        if leaves.size == 0 or codes.size == 0:
            raise RuntimeError("Weighted Prüfer construction reached an invalid state")
        candidate_scores = scores[np.ix_(leaves, codes)]
        flat_index = int(np.argmax(candidate_scores))
        leaf_index, code_index = np.unravel_index(flat_index, candidate_scores.shape)
        leaf = int(leaves[leaf_index])
        code = int(codes[code_index])
        graph.add_edge(leaf, code)
        active[leaf] = False
        occurrences[code] -= 1

    remaining = np.flatnonzero(active)
    if remaining.size != 2:
        raise RuntimeError(
            f"Weighted Prüfer construction ended with {remaining.size} active nodes"
        )
    graph.add_edge(int(remaining[0]), int(remaining[1]))
    if graph.number_of_edges() != node_count - 1 or not nx.is_tree(graph):
        raise RuntimeError("Weighted Prüfer decoder failed its tree invariant")
    realized = np.array([graph.degree(node) for node in range(node_count)])
    if not np.array_equal(realized, target):
        raise RuntimeError(
            f"Weighted Prüfer degree mismatch: target={target}, realized={realized}"
        )
    return graph


def decode_planar_constrained(
    edge_scores,
    target_degrees,
    target_edges: int,
    noise_scale: float = 0.0,
) -> nx.Graph:
    """Build a connected planar graph with exactly ``target_edges`` edges."""
    target = np.asarray(target_degrees, dtype=np.int64).reshape(-1)
    node_count = int(target.size)
    target_edges = int(target_edges)
    if node_count < 3:
        raise ValueError("Planar constrained decoding requires at least three nodes")
    planar_max = 3 * node_count - 6
    if target_edges > planar_max:
        raise ValueError(
            f"A simple planar graph with n={node_count} cannot have "
            f"E={target_edges} > 3n-6={planar_max}"
        )
    if target_edges < node_count - 1:
        raise ValueError(
            f"A connected graph with n={node_count} needs at least n-1 edges; "
            f"got E={target_edges}"
        )
    if target.shape[0] != node_count or int(target.sum()) != 2 * target_edges:
        raise ValueError(
            f"Planar target degrees must sum to 2E={2 * target_edges}; "
            f"got {target.sum()}"
        )

    scores = _noisy_symmetric_scores(edge_scores, noise_scale)
    if scores.shape[0] != node_count:
        raise ValueError(
            f"Planar scores/degree size mismatch: {scores.shape[0]} vs {node_count}"
        )
    finite_upper = scores[np.triu_indices(node_count, 1)]
    score_scale = max(float(np.std(finite_upper)), 1e-3)
    target_scale = max(float(target.max()), 1.0)

    # First create a maximum-weight spanning tree.  The target-degree prior
    # breaks locally ambiguous logits in favour of the sampled degree shape.
    all_edges = []
    for u in range(node_count):
        for v in range(u + 1, node_count):
            prior = 0.35 * score_scale * (target[u] + target[v]) / (2.0 * target_scale)
            all_edges.append((float(scores[u, v] + prior), u, v))
    all_edges.sort(key=lambda item: (-item[0], item[1], item[2]))

    parent = list(range(node_count))
    rank = [0] * node_count

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for _, u, v in all_edges:
        if union(u, v):
            graph.add_edge(u, v)
            if graph.number_of_edges() == node_count - 1:
                break
    if not nx.is_tree(graph):
        raise RuntimeError("Planar backbone construction did not produce a tree")

    # Add edges according to logits and the current target-degree deficit.
    # Rejected non-edges can be removed permanently: adding more edges cannot
    # make a previously non-planar supergraph planar again.
    candidates = {(u, v) for _, u, v in all_edges if not graph.has_edge(u, v)}
    while graph.number_of_edges() < target_edges:
        degrees = np.array([graph.degree(node) for node in range(node_count)])
        deficits = np.maximum(target - degrees, 0) / target_scale
        ranked = sorted(
            candidates,
            key=lambda edge: (
                -(scores[edge[0], edge[1]]
                  + 0.75 * score_scale * (deficits[edge[0]] + deficits[edge[1]])),
                edge[0],
                edge[1],
            ),
        )
        accepted = False
        for u, v in ranked:
            graph.add_edge(u, v)
            if nx.check_planarity(graph)[0]:
                candidates.remove((u, v))
                accepted = True
                break
            graph.remove_edge(u, v)
            candidates.remove((u, v))
        if not accepted:
            raise RuntimeError(
                f"Could not augment planar graph to E={target_edges}; "
                f"stopped at {graph.number_of_edges()}"
            )

    if (
        graph.number_of_nodes() != node_count
        or graph.number_of_edges() != target_edges
        or not nx.is_connected(graph)
        or not nx.check_planarity(graph)[0]
    ):
        raise RuntimeError("Planar decoder failed its structural invariant")
    return graph


def decode_samples_to_graphs(
    args,
    adj_probs,
    node_logits,
    stats_pred,
    degree_pred=None,
    edge_logits_raw=None,
    profile=None,
    sampled_profile_hist=None,
):
    """Convert decoder outputs into NetworkX graphs.
    
    Top-E decoding uses a predicted edge budget and optional connectivity
    repair. Threshold decoding uses probability 0.5 and keeps the largest
    connected component. Dataset configurations select structural constraints."""
    adj_probs_cpu = adj_probs.detach().cpu()
    node_probs = torch.sigmoid(node_logits.detach().cpu())
    stats_pred_cpu = stats_pred.detach().cpu()
    degree_pred_cpu = degree_pred.detach().cpu() if degree_pred is not None else None
    edge_logits_cpu = edge_logits_raw.detach().cpu() if edge_logits_raw is not None else None
    sampled_profile_hist_cpu = (
        sampled_profile_hist.detach().cpu()
        if sampled_profile_hist is not None
        else None
    )
    max_nodes = adj_probs_cpu.shape[-1]

    mode = getattr(args, "decode_mode", "topE")
    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    if structure_constraint not in {"none", "planar", "tree"}:
        raise ValueError(
            "structure_constraint must be one of none, planar, tree; "
            f"got {structure_constraint!r}"
        )
    if structure_constraint != "none" and bool(getattr(args, "directed", False)):
        raise ValueError("Planar/tree constrained decoding only supports undirected graphs")
    if (
        structure_constraint != "none"
        and sampled_profile_hist_cpu is not None
        and sampled_profile_hist_cpu.shape[0] != adj_probs_cpu.shape[0]
    ):
        raise ValueError(
            "sampled_profile_hist batch does not match decoder outputs: "
            f"hist={sampled_profile_hist_cpu.shape[0]}, "
            f"outputs={adj_probs_cpu.shape[0]}"
        )
    use_repair = bool(getattr(args, "connectivity_repair", True)) and mode == "topE"
    rel_tol = float(getattr(args, "topE_tolerance", 0.10))
    abs_tol = int(getattr(args, "topE_abs_tol", 2))
    repair_rel = float(getattr(args, "edge_budget_rel_tol", 0.05))
    repair_abs = int(getattr(args, "edge_budget_abs_tol", 2))

    n_min = profile.n_min if profile is not None else 1
    n_max = profile.n_max if profile is not None else max_nodes

    generated_graphs = []
    for i in range(adj_probs_cpu.shape[0]):
        if structure_constraint != "none":
            n_pred = int(
                torch.clamp(
                    torch.round(stats_pred_cpu[i, 0]), n_min, n_max
                ).item()
            )
            if n_pred < 2:
                raise ValueError(
                    f"Constrained decoding needs at least two nodes, got {n_pred}"
                )
            topk = torch.topk(node_probs[i], k=n_pred).indices
            if edge_logits_cpu is not None:
                sub_scores = edge_logits_cpu[i][topk][:, topk].numpy()
            else:
                sub_prob = np.clip(
                    adj_probs_cpu[i][topk][:, topk].numpy(), 1e-6, 1.0 - 1e-6
                )
                sub_scores = np.log(sub_prob / (1.0 - sub_prob))
            sub_scores = 0.5 * (sub_scores + sub_scores.T)

            if degree_pred_cpu is not None:
                predicted_degrees = degree_pred_cpu[i][topk].numpy()
            else:
                predicted_degrees = adj_probs_cpu[i][topk][:, topk].sum(dim=-1).numpy()
            if sampled_profile_hist_cpu is not None:
                profile_histogram = sampled_profile_hist_cpu[i].numpy()
            elif profile is not None and profile.hist_mean is not None:
                profile_histogram = profile.hist_mean.numpy()
            else:
                profile_histogram = np.ones(32, dtype=np.float64)

            blend = float(getattr(args, "degree_profile_blend", 0.0))
            noise_scale = float(getattr(args, "constraint_noise_scale", 0.0))
            if structure_constraint == "tree":
                target_edges = n_pred - 1
                target_degrees = blend_degree_targets(
                    predicted_degrees,
                    profile_histogram,
                    blend=blend,
                    total_degree=2 * target_edges,
                    min_degree=1,
                )
                graph = decode_tree_constrained(
                    sub_scores, target_degrees, noise_scale=noise_scale
                )
            else:
                target_edges = int(round(float(stats_pred_cpu[i, 1].item())))
                if profile is not None:
                    target_edges = max(profile.m_min, min(profile.m_max, target_edges))
                planar_max = 3 * n_pred - 6
                if target_edges > planar_max:
                    raise ValueError(
                        f"Sampled Planar E={target_edges} exceeds 3n-6={planar_max}"
                    )
                target_degrees = blend_degree_targets(
                    predicted_degrees,
                    profile_histogram,
                    blend=blend,
                    total_degree=2 * target_edges,
                    min_degree=1,
                )
                graph = decode_planar_constrained(
                    sub_scores,
                    target_degrees,
                    target_edges=target_edges,
                    noise_scale=noise_scale,
                )
            generated_graphs.append(graph)
            continue

        if mode == "threshold":
            binary_adj, _ = _decode_one_threshold(
                adj_probs_cpu[i], node_probs[i], stats_pred_cpu[i],
                args.directed, max_nodes,
            )
            G = nx.from_numpy_array(binary_adj)
            G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() == 0:
                continue
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            generated_graphs.append(G)
            continue

        # ---- topE path ----
        degree_pred_i = degree_pred_cpu[i] if degree_pred_cpu is not None else None
        binary_adj, topk = _decode_one_topE(
            adj_probs_cpu[i], node_probs[i], stats_pred_cpu[i], degree_pred_i,
            args.directed, max_nodes, n_min, n_max,
            rel_tol, abs_tol,
        )
        n = binary_adj.shape[0]
        if n == 0:
            continue
        G = nx.from_numpy_array(binary_adj)

        target_E = int(round(stats_pred_cpu[i, 1].item()))
        max_E = n * (n - 1) // 2
        target_E = max(0, min(max_E, target_E))

        if use_repair and n >= 2:
            # Build symmetric logit matrix over top-n nodes for repair decisions.
            if edge_logits_cpu is not None:
                sub_logits = edge_logits_cpu[i][topk][:, topk].numpy()
            else:
                # Fall back to log-odds proxy.
                p = np.clip(binary_adj * 0 + 0.5, 1e-6, 1 - 1e-6)  # neutral
                sub_logits = np.log(p / (1 - p))
                # Use adj_probs as a soft signal.
                ap = adj_probs_cpu[i][topk][:, topk].numpy()
                sub_logits = np.log(np.clip(ap, 1e-6, 1 - 1e-6) /
                                    np.clip(1 - ap, 1e-6, 1 - 1e-6))
            if not args.directed:
                sub_logits = 0.5 * (sub_logits + sub_logits.T)
            G = _connectivity_repair(G, sub_logits, target_E, repair_rel, repair_abs)
        else:
            # Retain the largest connected component when repair is disabled.
            G.remove_edges_from(nx.selfloop_edges(G))
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() == 0:
                continue
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        if G.number_of_nodes() > 0:
            generated_graphs.append(G)
    return generated_graphs


def sample_and_decode(
    args,
    flow_model,
    decoder,
    cond_bank,
    num_samples,
    device,
    profile=None,
    condition_profile_bank=None,
):
    """Sample conditional latent vectors and decode graph candidates.
    
    Jointly sample learned conditions and structural statistics from the
    training bank. Optional noise perturbs the learned condition vector.
    Decoding applies the configured edge budget and connectivity constraints."""
    print("\n" + "=" * 60)
    print("Phase 3: Sampling and Decoding")
    print("=" * 60)

    flow_model.eval()
    decoder.eval()
    cond_bank = cond_bank.to(device)
    n_bank = cond_bank.shape[0]

    cond_noise_std = float(getattr(args, "cond_noise_std", 0.05))
    use_struct = bool(getattr(args, "use_struct_cond", True)) and profile is not None

    if use_struct:
        struct_dim = profile.profile_stat_dim   # 8 + 32
        struct_cond_dim_proj = cond_bank.shape[1] - 0  # not used directly here
        # Pick indices into bank; pair with stats from profile.sample_joint using same indices
        idx = torch.randint(0, n_bank, (num_samples,), device=device)
        sampled_cond = cond_bank[idx].clone()
        # Pull standardized scalars + raw hist for those same training indices.
        if condition_profile_bank is None:
            scalar_bank = profile.scalar_bank
            histogram_bank = profile.hist_bank
        else:
            scalar_bank, histogram_bank = condition_profile_bank
            if scalar_bank.shape[0] != n_bank or histogram_bank.shape[0] != n_bank:
                raise ValueError(
                    "Aligned condition/profile banks must match cond_bank: "
                    f"cond={n_bank}, scalar={scalar_bank.shape[0]}, "
                    f"hist={histogram_bank.shape[0]}"
                )
        std_scalar = scalar_bank[idx.cpu()].to(device)
        hist = histogram_bank[idx.cpu()].to(device)
        # Add small Gaussian noise to the learned-cond base only (first cond_dim entries).
        # We don't know cond_dim split here directly — apply tiny noise to the whole vector.
        if cond_noise_std > 0:
            sampled_cond = sampled_cond + cond_noise_std * torch.randn_like(sampled_cond)
        joint_profile_vec = torch.cat([std_scalar, hist], dim=-1)
    else:
        idx = torch.randint(0, n_bank, (num_samples,), device=device)
        sampled_cond = cond_bank[idx]
        if cond_noise_std > 0:
            sampled_cond = sampled_cond + cond_noise_std * torch.randn_like(sampled_cond)
        joint_profile_vec = None

    with torch.no_grad():
        integrator = getattr(args, 'flow_integrator', 'euler')
        guidance_scale = getattr(args, 'flow_guidance_scale', 1.2)
        if integrator == 'heun':
            samples_klein = flow_model.sample_heun(sampled_cond, guidance_scale=guidance_scale)
        else:
            samples_klein = flow_model.sample(sampled_cond, guidance_scale=guidance_scale)

        samples_tangent = Klein().logmap0(samples_klein, c=1.0)
        edge_logits, node_logits, stats_pred, degree_pred = decoder(
            samples_tangent.to(device), sampled_cond.to(device)
        )
        adj_probs = torch.sigmoid(edge_logits)

        # If we have a profile-sampled stats vector, override the decoder's
        # predicted (n, m) with the bank-sampled values to give explicit control.
        if joint_profile_vec is not None:
            # First two entries of std_scalar are standardized (n, m). Destandardize.
            raw_nm = profile.destandardize_scalar(std_scalar)[:, :2]
            stats_pred = stats_pred.clone()
            stats_pred[:, 0] = raw_nm[:, 0]
            stats_pred[:, 1] = raw_nm[:, 1]

    generated_graphs = decode_samples_to_graphs(
        args, adj_probs, node_logits, stats_pred,
        degree_pred=degree_pred,
        edge_logits_raw=edge_logits,
        profile=profile,
        sampled_profile_hist=(hist if use_struct else None),
    )
    print(f"Generated {len(generated_graphs)} valid graphs")
    return generated_graphs, samples_klein, sampled_cond


def clean_graphs_like_evaluate(graph_real, graph_pred):
    """Match generated graph sizes to the real set."""
    graph_real = list(graph_real)
    graph_pred = list(graph_pred)

    random.shuffle(graph_real)
    random.shuffle(graph_pred)

    if len(graph_real) == 0 or len(graph_pred) == 0:
        return graph_real, graph_pred

    real_graph_len = np.array([len(g) for g in graph_real])
    pred_graph_len = np.array([len(g) for g in graph_pred])

    pred_graph_new = []
    for value in real_graph_len:
        pred_idx = (np.abs(pred_graph_len - value)).argmin()
        pred_graph_new.append(graph_pred[pred_idx])

    return graph_real, pred_graph_new


def _adjacency_list_to_graphs(adjacencies):
    graphs = []
    for adjacency in adjacencies:
        graph = nx.from_numpy_array(adjacency.toarray())
        graphs.append(normalize_graph(graph))
    return graphs


def _evaluate_vun_ratio(
    args, generated_graphs, train_core_adj, test_list_adj, graph_save_path
):
    """Run final SimGFM evaluation without additional graph cleanup."""
    train_graphs = _adjacency_list_to_graphs(train_core_adj)
    test_graphs = _adjacency_list_to_graphs(test_list_adj)
    generated = [normalize_graph(graph) for graph in generated_graphs]
    results = evaluate_external_benchmark(
        args.dataset,
        generated,
        train_graphs,
        test_graphs,
    )

    print("\n" + "-" * 54)
    print("SimGFM V.U.N. / Ratio Evaluation Results:")
    print("-" * 54)
    print(f"  V.U.N.:             {results['vun']:.6f}")
    print(f"  Valid:              {results['frac_valid']:.6f}")
    print(f"  Unique:             {results['frac_unique']:.6f}")
    print(f"  Novel:              {results['frac_non_iso']:.6f}")
    print(f"  Unique + Novel:     {results['frac_unique_non_iso']:.6f}")
    for key in ("degree", "clustering", "orbit", "spectre", "wavelet"):
        ratio_key = f"{key}_ratio"
        if ratio_key in results:
            print(f"  {ratio_key:<19} {results[ratio_key]:.6f}")
    print(f"  Average Ratio:      {results['average_ratio']:.6f}")
    print("-" * 54)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results_file = Path(graph_save_path) / "evaluation_results.txt"
    with open(results_file, "w", encoding="utf-8") as handle:
        handle.write("=" * 70 + "\n")
        handle.write("Klein Flow SimGFM Benchmark Evaluation\n")
        handle.write("=" * 70 + "\n\n")
        handle.write(f"Dataset: {args.dataset}\n")
        handle.write("Metric profile: vun_ratio\n")
        handle.write("Split: benchmark train=128 val=32 test=40\n")
        postprocessing = (
            f"{getattr(args, 'structure_constraint', 'none')} constrained decoding"
            if str(getattr(args, "structure_constraint", "none")).lower()
            in {"planar", "tree"}
            else "Top-E decoding with connectivity repair"
        )
        handle.write(
            f"Generation postprocessing: {postprocessing}, profile reranking\n"
        )
        handle.write(f"Date: {timestamp}\n\n")
        handle.write("--- Metrics ---\n")
        for key, value in results.items():
            handle.write(f"{key}: {float(value):.10f}\n")
        handle.write("\n--- Graph Statistics ---\n")
        handle.write(f"Train graphs: {len(train_graphs)}\n")
        handle.write(f"Test graphs: {len(test_graphs)}\n")
        handle.write(f"Generated graphs: {len(generated)}\n")
        handle.write(
            f"Average nodes test/generated: "
            f"{np.mean([g.number_of_nodes() for g in test_graphs]):.2f} / "
            f"{np.mean([g.number_of_nodes() for g in generated]):.2f}\n"
        )
        handle.write(
            f"Average edges test/generated: "
            f"{np.mean([g.number_of_edges() for g in test_graphs]):.2f} / "
            f"{np.mean([g.number_of_edges() for g in generated]):.2f}\n"
        )

    logging.info("V.U.N./Ratio results: %s", results)
    _write_metrics_json(args, results, graph_save_path, timestamp)
    return results


def evaluate_and_save_results(
    args, generated_graphs, test_list_adj, graph_save_path, train_core_adj=None
):
    """Evaluate generated graphs and save results."""
    print("\n" + "=" * 60)
    print("Phase 4: Evaluation and Results")
    print("=" * 60)

    if evaluation_profile(args.dataset) == "vun_ratio":
        if train_core_adj is None:
            raise ValueError("V.U.N. novelty requires the fixed training split")
        return _evaluate_vun_ratio(
            args,
            generated_graphs,
            train_core_adj,
            test_list_adj,
            graph_save_path,
        )

    test_graphs = []
    for adj in test_list_adj:
        G = nx.from_numpy_array(adj.toarray())
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() > 0:
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            test_graphs.append(G)

    if len(test_graphs) == 0 or len(generated_graphs) == 0:
        raise ValueError(
            f"Invalid evaluation inputs: test_graphs={len(test_graphs)}, generated_graphs={len(generated_graphs)}"
        )

    eval_test_graphs, eval_generated_graphs = clean_graphs_like_evaluate(test_graphs, generated_graphs)

    mmd_degree = degree_stats(eval_test_graphs, eval_generated_graphs, compute_emd=True)
    mmd_clustering = clustering_stats(eval_test_graphs, eval_generated_graphs, compute_emd=True)
    mmd_spectral = spectral_stats(eval_test_graphs, eval_generated_graphs, compute_emd=True)
    avg_mmd = (mmd_degree + mmd_clustering + mmd_spectral) / 3.0

    print("\n" + "-" * 40)
    print("MMD Evaluation Results:")
    print("-" * 40)
    print(f"  Degree MMD:      {mmd_degree:.6f}")
    print(f"  Clustering MMD:  {mmd_clustering:.6f}")
    print(f"  Spectral MMD:    {mmd_spectral:.6f}")
    print(f"  Average MMD:     {avg_mmd:.6f}")
    print("-" * 40)

    avg_nodes_test = np.mean([G.number_of_nodes() for G in eval_test_graphs])
    avg_nodes_gen = np.mean([G.number_of_nodes() for G in eval_generated_graphs])
    avg_edges_test = np.mean([G.number_of_edges() for G in eval_test_graphs])
    avg_edges_gen = np.mean([G.number_of_edges() for G in eval_generated_graphs])

    results_file = graph_save_path + 'mmd_results.txt'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(results_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Klein Flow Matching Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Date: {timestamp}\n\n")
        f.write("--- Model Configuration ---\n")
        f.write(f"Graph Embedding Dim: {args.graphEmDim}\n")
        f.write(f"Condition Dim: {getattr(args, 'cond_dim', 64)}\n")
        f.write(f"Decoder Node Dim: {getattr(args, 'decoder_node_dim', 128)}\n")
        f.write(f"DiT Hidden Dim: {getattr(args, 'dit_hidden_dim', 512)}\n")
        f.write(f"DiT Num Heads: {getattr(args, 'dit_num_heads', 8)}\n")
        f.write(f"DiT Num Layers: {getattr(args, 'dit_num_layers', 6)}\n")
        f.write(f"Flow Steps: {getattr(args, 'flow_steps', 200)}\n\n")
        f.write("--- MMD Metrics on Test Set ---\n")
        f.write(f"Degree MMD:      {mmd_degree:.6f}\n")
        f.write(f"Clustering MMD:  {mmd_clustering:.6f}\n")
        f.write(f"Spectral MMD:    {mmd_spectral:.6f}\n")
        f.write(f"Average MMD:     {avg_mmd:.6f}\n\n")
        f.write("--- Graph Statistics ---\n")
        f.write(f"Test Set graphs: {len(eval_test_graphs)}\n")
        f.write(f"Generated graphs: {len(eval_generated_graphs)}\n")
        f.write(f"Avg nodes test/gen: {avg_nodes_test:.2f} / {avg_nodes_gen:.2f}\n")
        f.write(f"Avg edges test/gen: {avg_edges_test:.2f} / {avg_edges_gen:.2f}\n")

    logging.info(
        f"Degree MMD: {mmd_degree}, Clustering MMD: {mmd_clustering}, "
        f"Spectral MMD: {mmd_spectral}, Average MMD: {avg_mmd}"
    )

    results = {
        'mmd_degree': mmd_degree,
        'mmd_clustering': mmd_clustering,
        'mmd_spectral': mmd_spectral,
        'avg_mmd': avg_mmd
    }
    _write_metrics_json(args, results, graph_save_path, timestamp)
    return results


def klein_graphtask(args):
    """Run graph autoencoder training, Flow Matching, sampling, and evaluation."""
    from flow_klein.paths import prepare_training_args
    prepare_training_args(args, "structural")
    if is_benchmark_dataset(args.dataset):
        args.dataset = normalize_benchmark_name(args.dataset)
    training_seed = int(getattr(args, "seed", 0))
    np.random.seed(training_seed)
    random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.graph_save_path is None:
        args.graph_save_path = str(OUTPUT_ROOT / args.dataset) + '/' 
    graph_save_path = args.graph_save_path
    Path(graph_save_path).mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=graph_save_path + 'klein_graphtask.log',
        filemode='w',
        level=logging.INFO
    )

    print("=" * 60)
    print("KleinFlow: Graph Generation with Conditional Flow Matching")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Training seed: {training_seed}")
    if is_benchmark_dataset(args.dataset):
        print("Split: benchmark train=128 val=32 test=40")
    else:
        print(f"Outer split seed: {OUTER_SPLIT_SEED}")
        print(f"Validation split seed: {int(getattr(args, 'split_seed', 1432))}")
    print(f"Save path: {graph_save_path}")
    logging.info(f"Klein GraphTask Settings: {args}")

    device = torch.device(args.device if torch.cuda.is_available() and args.UseGPU else "cpu")
    print(f"Device: {device}")

    # apply dataset-specific defaults for any flag the
    # user did NOT override on the CLI.
    try:
        from flow_klein.config.structural import apply_klein_dataset_defaults
        apply_klein_dataset_defaults(args)
    except Exception as exc:
        print(f"[config] dataset-defaults hook skipped ({exc!r})")

    if evaluation_profile(args.dataset) == "vun_ratio":
        print("\nPreflighting ORCA, PyGSP, and benchmark dependencies...")
        preflight_metric_dependencies(args.dataset)

    print("\nLoading data...")
    list_graphs, list_test_graphs, val_adj, test_list_adj, train_core_adj, val_graphs = load_data(args)
    print(f"Training graphs (train-core): {len(list_graphs.list_adjs)}")
    print(f"Validation graphs:            {len(val_adj)}")
    print(f"Test graphs:                  {len(test_list_adj)}")
    print(f"Max nodes: {list_graphs.max_num_nodes}")
    print(f"Feature dim: {list_graphs.feature_size}")

    (
        encoder_model,
        klein_embeddings,
        cond_codes,
        profile,
        aligned_profile_bank,
    ) = train_klein_encoder(
        args, list_graphs, val_adj, train_core_adj, val_graphs, device,
    )
    structure_constraint = str(
        getattr(args, "structure_constraint", "none")
    ).strip().lower()
    constraint_profile_bank = (
        aligned_profile_bank
        if structure_constraint in {"planar", "tree"}
        else None
    )
    flow_model = train_klein_flow_matching(
        args, klein_embeddings, cond_codes, device,
        encoder_model=encoder_model, val_adj=val_adj, profile=profile,
        condition_profile_bank=constraint_profile_bank,
        train_core_adj=train_core_adj,
    )

    decoder = encoder_model.decoder.to(device)
    decoder.eval()

    # candidate oversampling + train-profile reranking.
    n_target = len(test_list_adj)
    candidate_multiplier = max(1, int(getattr(args, "candidate_multiplier", 3)))
    profile_select = bool(getattr(args, "profile_select", True))
    n_candidates = candidate_multiplier * n_target

    generated_graphs, samples_klein, sampled_cond = sample_and_decode(
        args,
        flow_model,
        decoder,
        cond_codes,
        n_candidates,
        device,
        profile=profile,
        condition_profile_bank=constraint_profile_bank,
    )
    if profile_select and candidate_multiplier > 1 and profile is not None and generated_graphs:
        before = len(generated_graphs)
        prefer_unique = bool(getattr(args, "unique_candidate_select", False))
        generated_graphs = rerank_candidates_by_profile(
            generated_graphs,
            profile,
            n_target,
            prefer_unique=prefer_unique,
        )
        unique_note = " with generated-candidate uniqueness preference" if prefer_unique else ""
        print(
            f"[rerank] kept {len(generated_graphs)}/{before} candidates "
            f"by train-profile distance{unique_note}"
        )
    else:
        generated_graphs = generated_graphs[:n_target]

    if structure_constraint in {"planar", "tree"}:
        if len(generated_graphs) != n_target:
            raise RuntimeError(
                f"Constrained generation produced {len(generated_graphs)} graphs; "
                f"expected exactly {n_target}"
            )
        validity_function = (
            is_planar_graph if structure_constraint == "planar" else is_tree_graph
        )
        invalid_indices = [
            index
            for index, graph in enumerate(generated_graphs)
            if not validity_function(graph)
        ]
        profile_shape_violations = []
        for index, graph in enumerate(generated_graphs):
            node_count = graph.number_of_nodes()
            edge_count = graph.number_of_edges()
            node_ok = profile.n_min <= node_count <= profile.n_max
            if structure_constraint == "tree":
                edge_ok = edge_count == node_count - 1
            else:
                edge_ok = profile.m_min <= edge_count <= profile.m_max
            if not node_ok or not edge_ok:
                profile_shape_violations.append(index)
        if invalid_indices:
            raise RuntimeError(
                f"{structure_constraint} constrained generation produced invalid "
                f"graphs at indices {invalid_indices[:10]}"
            )
        if profile_shape_violations:
            raise RuntimeError(
                f"{structure_constraint} constrained generation violated the "
                f"training profile at indices {profile_shape_violations[:10]}"
            )

    import pickle
    with open(graph_save_path + 'generated_graphs.pkl', 'wb') as f:
        pickle.dump(generated_graphs, f)
    torch.save(samples_klein, graph_save_path + 'generated_klein_samples.pt')
    torch.save(sampled_cond, graph_save_path + 'generated_cond_samples.pt')

    results = evaluate_and_save_results(
        args,
        generated_graphs,
        test_list_adj,
        graph_save_path,
        train_core_adj=train_core_adj,
    )

    if structure_constraint in {"planar", "tree"}:
        non_finite = [
            key
            for key, value in results.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and not np.isfinite(float(value))
        ]
        if non_finite:
            raise RuntimeError(
                f"Constrained benchmark returned non-finite metrics: {non_finite}"
            )
        if float(results.get("frac_valid", 0.0)) != 1.0:
            raise RuntimeError(
                f"Constrained benchmark frac_valid must be 1.0, got "
                f"{results.get('frac_valid')!r}"
            )

    print("\n" + "=" * 60)
    print("Klein GraphTask Complete!")
    print("=" * 60)
    if evaluation_profile(args.dataset) == "vun_ratio":
        print(
            f"Final V.U.N.: {results['vun']:.6f}; "
            f"Average Ratio: {results['average_ratio']:.6f}"
        )
    else:
        print(f"Final Average MMD: {results['avg_mmd']:.6f}")

    torch.cuda.empty_cache()
    return results
