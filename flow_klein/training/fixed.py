from flow_klein.paths import OUTPUT_ROOT
"""
Klein GraphTask V2: graph generation with structural encoder/decoder upgrades
and conditional Klein flow matching.
"""

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

from flow_klein.data.fixed import Datasets, list_graph_loader, data_split, BFS
from flow_klein.data.benchmarks_fixed import (
    is_benchmark_dataset,
    load_benchmark_splits,
    normalize_benchmark_name,
)
from flow_klein.models.kernels import kernel
from flow_klein.models.fixed import KleinEncoder, KleinGraphVAE, MaskedGraphDecoder
from flow_klein.models.flow_matching import KleinFlowMatching
from flow_klein.geometry.klein import Klein
from flow_klein.evaluation.spectre import degree_stats, clustering_stats, spectral_stats
from flow_klein.evaluation.fixed import (
    evaluate_external_benchmark,
    evaluation_profile,
    preflight_metric_dependencies,
)


OUTER_SPLIT_SEED = 123
USES_INNER_VALIDATION_SPLIT = False




def _write_metrics_json(args, results, graph_save_path, timestamp):
    from flow_klein.training.reporting import write_metrics_json
    write_metrics_json(args, results, graph_save_path, timestamp,
                       OUTER_SPLIT_SEED, USES_INNER_VALIDATION_SPLIT)



def load_data(args):
    """Load and prepare graph dataset."""
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
        train_adj = list(splits["train"])
        val_adj = list(splits["val"])
        test_list_adj = list(splits["test"])
        if args.bfsOrdering:
            train_adj = BFS(train_adj)
            val_adj = BFS(val_adj)
            test_list_adj = BFS(test_list_adj)

        all_adjs = train_adj + val_adj + test_list_adj
        max_size = max(adjacency.shape[0] for adjacency in all_adjs)
        train_x = [None] * len(train_adj)
        test_x = [None] * len(test_list_adj)
        list_graphs = Datasets(
            train_adj,
            self_for_none,
            train_x,
            None,
            Max_num=max_size,
            set_diag_of_isol_Zer=False,
            **dataset_kwargs,
        )
        list_test_graphs = Datasets(
            test_list_adj,
            self_for_none,
            test_x,
            None,
            Max_num=max_size,
            set_diag_of_isol_Zer=False,
            **dataset_kwargs,
        )
        return list_graphs, list_test_graphs, val_adj, test_list_adj, train_adj

    list_adj, list_x, list_label = list_graph_loader(
        dataset, return_labels=True, shuffle=False
    )

    if args.bfsOrdering:
        list_adj = BFS(list_adj)

    if len(list_adj) == 1:
        test_list_adj = list_adj.copy()
        train_adj = list_adj
        list_graphs = Datasets(list_adj, self_for_none, list_x, None, **dataset_kwargs)
        list_test_graphs = list_graphs
        val_adj = list_adj
    else:
        max_size = None
        list_adj, test_list_adj, list_x_train, list_x_test, _, list_label_test = data_split(
            list_adj, list_x, list_label, split_seed=OUTER_SPLIT_SEED
        )
        train_adj = list_adj
        val_adj = list_adj[:int(len(test_list_adj))]
        list_graphs = Datasets(
            list_adj, self_for_none, list_x_train, list_label,
            Max_num=max_size, set_diag_of_isol_Zer=False, **dataset_kwargs
        )
        list_test_graphs = Datasets(
            test_list_adj, self_for_none, list_x_test, list_label_test,
            Max_num=list_graphs.max_num_nodes, set_diag_of_isol_Zer=False, **dataset_kwargs
        )

    return list_graphs, list_test_graphs, val_adj, test_list_adj, train_adj


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
    """Compute graph-level targets on padded adjacency batches."""
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


def compute_kernel_matching_loss(reconstructed_adj, target_kernel_val, kernel_model, alpha=0.1):
    """Auxiliary kernel matching loss with reduced weight."""
    if kernel_model is None or target_kernel_val is None:
        return reconstructed_adj.new_tensor(0.0)
    reconstructed_kernel_val = kernel_model(reconstructed_adj)
    kernel_loss = reconstructed_adj.new_tensor(0.0)
    for generated, target in zip(reconstructed_kernel_val, target_kernel_val):
        kernel_loss = kernel_loss + F.smooth_l1_loss(generated, target.to(generated.device))
    return alpha * kernel_loss


def compute_vae_loss_v2(
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
    kernel_weight=0.1,
    node_weight=0.25,
    stats_weight=0.1,
    kl_weight=0.05,
):
    """Compute V2 reconstruction loss with node masking and auxiliary stats."""
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

    total_loss = edge_loss + node_weight * node_loss + stats_weight * stats_loss + kl_weight * kl_loss + kernel_loss

    predictions = (adj_probs > 0.5).float()
    acc = ((predictions == target_adj).float() * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)

    return {
        "total": total_loss,
        "edge": edge_loss,
        "node": node_loss,
        "stats": stats_loss,
        "kl": kl_loss,
        "kernel": kernel_loss,
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
def collect_encoder_outputs(args, model, list_graphs, device):
    """Collect deterministic Klein embeddings and conditional codes."""
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

        h_klein, _, _, cond_code, _ = model.encode(
            org_adj_dgl, x_s, batch_size, node_mask=node_mask
        )
        all_embeddings.append(h_klein.cpu())
        all_cond_codes.append(cond_code.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_cond_codes, dim=0)


def train_klein_encoder(args, list_graphs, val_adj, device):
    """Train the upgraded Klein encoder/decoder stack."""
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

    hidden_dim = max(128, min(256, graph_em_dim * 4))
    encoder = KleinEncoder(
        in_feature_dim=in_feature_dim,
        hidden_layers=[hidden_dim] * encoder_blocks,
        graph_latent_dim=graph_em_dim,
        dropout=args.dropout,
        cond_dim=cond_dim,
        encoder_blocks=encoder_blocks,
        input_proj_dim=64,
    )

    decoder = MaskedGraphDecoder(
        latent_dim=graph_em_dim,
        cond_dim=cond_dim,
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

    min_loss = float('inf')
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

            org_adj_dgl = prepare_batch_graphs(org_adj, device)

            model.train()
            optimizer.zero_grad()

            reconstructed_adj, samples, mean, log_std, cond_code, aux_preds, adj_logits = model(
                org_adj_dgl, x_s, batch_size, node_mask=node_mask
            )

            stats_target = compute_graph_batch_stats(subgraphs, node_mask)
            losses = compute_vae_loss_v2(
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
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(losses)

            total_loss_item = losses["total"].item()
            if total_loss_item < min_loss:
                min_loss = total_loss_item
                torch.save(model.state_dict(), graph_save_path + "klein_model_best.pt")

        scheduler.step()
        best_embeddings, best_cond_codes = collect_encoder_outputs(args, model, list_graphs, device)
        torch.save(best_embeddings, graph_save_path + f'{epoch}_klein_feat.pt')
        torch.save(best_cond_codes, graph_save_path + f'{epoch}_cond_feat.pt')

        if epoch_losses:
            edge = np.mean([loss["edge"].item() for loss in epoch_losses])
            node = np.mean([loss["node"].item() for loss in epoch_losses])
            stats = np.mean([loss["stats"].item() for loss in epoch_losses])
            kl = np.mean([loss["kl"].item() for loss in epoch_losses])
            acc = np.mean([loss["acc"].item() for loss in epoch_losses])
            total = np.mean([loss["total"].item() for loss in epoch_losses])
            print(
                f"Epoch {epoch + 1}/{epoch_number}, Total: {total:.6f}, Edge: {edge:.6f}, "
                f"Node: {node:.6f}, Stats: {stats:.6f}, KL: {kl:.6f}, Acc: {acc:.4f}"
            )

    best_model_path = graph_save_path + "klein_model_best.pt"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best encoder checkpoint from {best_model_path}")
        best_embeddings, best_cond_codes = collect_encoder_outputs(args, model, list_graphs, device)
        torch.save(best_embeddings, graph_save_path + "klein_feat_best.pt")
        torch.save(best_cond_codes, graph_save_path + "cond_feat_best.pt")

    torch.save(model.state_dict(), graph_save_path + "klein_encoder_final.pt")
    torch.save(model, graph_save_path + "klein_model_trained.pt")

    return model, best_embeddings, best_cond_codes


def train_klein_flow_matching(args, klein_embeddings, cond_codes, device):
    """Train conditional Klein Flow Matching."""
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

    tangent = flow_model.to_tangent_space(klein_embeddings.to(device))
    flow_model.set_latent_stats(
        tangent.mean(dim=0),
        tangent.std(dim=0).clamp_min(1e-4)
    )

    epochs = getattr(args, 'epoch_diff', 1000)
    batch_size = getattr(args, 'batchSize', 64)
    lr = getattr(args, 'lr_diff', 1e-4)

    optimizer = optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    klein_embeddings = klein_embeddings.to(device)
    cond_codes = cond_codes.to(device)
    num_samples = klein_embeddings.shape[0]

    print(f"Training Klein Flow Matching on {num_samples} samples, dim={dim}, cond_dim={cond_dim}")

    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        klein_embeddings = klein_embeddings[perm]
        cond_codes = cond_codes[perm]

        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch = klein_embeddings[i:i + batch_size]
            cond_batch = cond_codes[i:i + batch_size]

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

    torch.save(flow_model, graph_save_path + 'klein_flow_model.pt')
    torch.save(flow_model.state_dict(), graph_save_path + 'klein_flow_model_state.pt')
    return flow_model


def decode_samples_to_graphs(args, adj_probs, node_logits, stats_pred):
    """Convert decoder outputs into NetworkX graphs."""
    adj_probs = adj_probs.detach().cpu()
    node_probs = torch.sigmoid(node_logits.detach().cpu())
    stats_pred = stats_pred.detach().cpu()
    max_nodes = adj_probs.shape[-1]

    generated_graphs = []
    preserve_full_graph = evaluation_profile(args.dataset) != "degree_clustering_spectral"
    for i in range(adj_probs.shape[0]):
        predicted_nodes = int(torch.clamp(torch.round(stats_pred[i, 0]), 1, max_nodes).item())
        confident_nodes = int(node_probs[i].gt(0.5).sum().item())
        node_count = max(1, min(max_nodes, confident_nodes if confident_nodes > 0 else predicted_nodes))

        topk = torch.topk(node_probs[i], k=node_count).indices
        sub_adj = adj_probs[i][topk][:, topk]
        sub_adj.fill_diagonal_(0.0)

        binary_adj = (sub_adj > 0.5).float().numpy()
        if not args.directed:
            binary_adj = np.triu(binary_adj, 1)
            binary_adj = binary_adj + binary_adj.T

        G = nx.from_numpy_array(binary_adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        if preserve_full_graph:
            generated_graphs.append(G)
        else:
            G.remove_nodes_from(list(nx.isolates(G)))
            if G.number_of_nodes() > 0:
                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                generated_graphs.append(G)
    return generated_graphs


def sample_and_decode(args, flow_model, decoder, cond_bank, num_samples, device):
    """Sample from conditional Klein flow and decode to graphs."""
    print("\n" + "=" * 60)
    print("Phase 3: Sampling and Decoding")
    print("=" * 60)

    flow_model.eval()
    decoder.eval()
    cond_bank = cond_bank.to(device)

    choice = torch.randint(0, cond_bank.shape[0], (num_samples,), device=device)
    sampled_cond = cond_bank[choice]

    with torch.no_grad():
        integrator = getattr(args, 'flow_integrator', 'euler')
        guidance_scale = getattr(args, 'flow_guidance_scale', 1.2)
        if integrator == 'heun':
            samples_klein = flow_model.sample_heun(sampled_cond, guidance_scale=guidance_scale)
        else:
            samples_klein = flow_model.sample(sampled_cond, guidance_scale=guidance_scale)

        samples_tangent = Klein().logmap0(samples_klein, c=1.0)
        edge_logits, node_logits, stats_pred = decoder(samples_tangent.to(device), sampled_cond.to(device))
        adj_probs = torch.sigmoid(edge_logits)

    generated_graphs = decode_samples_to_graphs(args, adj_probs, node_logits, stats_pred)
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


def _adjacency_list_to_graphs(adjacencies, preserve_full_graph):
    graphs = []
    for adjacency in adjacencies:
        G = nx.from_numpy_array(adjacency.toarray())
        G.remove_edges_from(nx.selfloop_edges(G))
        if preserve_full_graph:
            graphs.append(G)
            continue
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() > 0:
            if not nx.is_connected(G):
                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            graphs.append(G)
    return graphs


def evaluate_and_save_results(
    args, generated_graphs, train_list_adj, test_list_adj, graph_save_path
):
    """Evaluate generated graphs and save results."""
    print("\n" + "=" * 60)
    print("Phase 4: Evaluation and Results")
    print("=" * 60)

    metric_profile = evaluation_profile(args.dataset)
    preserve_full_graph = metric_profile != "degree_clustering_spectral"
    test_graphs = _adjacency_list_to_graphs(test_list_adj, preserve_full_graph)
    train_graphs = _adjacency_list_to_graphs(train_list_adj, preserve_full_graph)

    if len(test_graphs) == 0 or len(generated_graphs) == 0:
        raise ValueError(
            f"Invalid evaluation inputs: test_graphs={len(test_graphs)}, generated_graphs={len(generated_graphs)}"
        )

    if metric_profile == "degree_clustering_spectral":
        eval_test_graphs, eval_generated_graphs = clean_graphs_like_evaluate(
            test_graphs, generated_graphs
        )
        mmd_degree = degree_stats(
            eval_test_graphs, eval_generated_graphs, compute_emd=True
        )
        mmd_clustering = clustering_stats(
            eval_test_graphs, eval_generated_graphs, compute_emd=True
        )
        mmd_spectral = spectral_stats(
            eval_test_graphs, eval_generated_graphs, compute_emd=True
        )
        results = {
            "mmd_degree": mmd_degree,
            "mmd_clustering": mmd_clustering,
            "mmd_spectral": mmd_spectral,
            "avg_mmd": (mmd_degree + mmd_clustering + mmd_spectral) / 3.0,
        }
    else:
        eval_test_graphs = test_graphs
        eval_generated_graphs = [nx.Graph(graph) for graph in generated_graphs]
        results = evaluate_external_benchmark(
            args.dataset,
            eval_generated_graphs,
            train_graphs,
            eval_test_graphs,
        )

    print("\n" + "-" * 40)
    print(f"Evaluation Results ({metric_profile}):")
    print("-" * 40)
    for key, value in results.items():
        print(f"  {key}: {float(value):.6f}")
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
        f.write(f"Metric profile: {metric_profile}\n")
        f.write(f"Date: {timestamp}\n\n")
        f.write("--- Model Configuration ---\n")
        f.write(f"Graph Embedding Dim: {args.graphEmDim}\n")
        f.write(f"Condition Dim: {getattr(args, 'cond_dim', 64)}\n")
        f.write(f"Decoder Node Dim: {getattr(args, 'decoder_node_dim', 128)}\n")
        f.write(f"DiT Hidden Dim: {getattr(args, 'dit_hidden_dim', 512)}\n")
        f.write(f"DiT Num Heads: {getattr(args, 'dit_num_heads', 8)}\n")
        f.write(f"DiT Num Layers: {getattr(args, 'dit_num_layers', 6)}\n")
        f.write(f"Flow Steps: {getattr(args, 'flow_steps', 200)}\n\n")
        f.write("--- Metrics on Test Set ---\n")
        for key, value in results.items():
            f.write(f"{key}: {float(value):.6f}\n")
        f.write("\n")
        f.write("--- Graph Statistics ---\n")
        f.write(f"Test Set graphs: {len(eval_test_graphs)}\n")
        f.write(f"Generated graphs: {len(eval_generated_graphs)}\n")
        f.write(f"Avg nodes test/gen: {avg_nodes_test:.2f} / {avg_nodes_gen:.2f}\n")
        f.write(f"Avg edges test/gen: {avg_edges_test:.2f} / {avg_edges_gen:.2f}\n")

    logging.info("Metric profile %s results: %s", metric_profile, results)
    _write_metrics_json(args, results, graph_save_path, timestamp)
    return results


def klein_graphtask(args):
    """Main entry point for Klein GraphTask V2."""
    from flow_klein.paths import prepare_training_args
    prepare_training_args(args, "fixed")
    training_seed = int(getattr(args, "seed", 0))
    np.random.seed(training_seed)
    random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if is_benchmark_dataset(args.dataset):
        args.dataset = normalize_benchmark_name(args.dataset)
        preflight_metric_dependencies(args.dataset)

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
    print("Klein GraphTask V2: Graph Generation with Conditional Klein Flow")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Training seed: {training_seed}")
    print(f"Outer split seed: {OUTER_SPLIT_SEED}")
    print(f"Save path: {graph_save_path}")
    logging.info(f"Klein GraphTask Settings: {args}")

    device = torch.device(args.device if torch.cuda.is_available() and args.UseGPU else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    list_graphs, list_test_graphs, val_adj, test_list_adj, train_list_adj = load_data(args)
    print(f"Training graphs: {len(list_graphs.list_adjs)}")
    print(f"Test graphs: {len(test_list_adj)}")
    print(f"Max nodes: {list_graphs.max_num_nodes}")
    print(f"Feature dim: {list_graphs.feature_size}")

    encoder_model, klein_embeddings, cond_codes = train_klein_encoder(args, list_graphs, val_adj, device)
    flow_model = train_klein_flow_matching(args, klein_embeddings, cond_codes, device)

    decoder = encoder_model.decoder.to(device)
    decoder.eval()

    num_samples = len(test_list_adj)
    generated_graphs, samples_klein, sampled_cond = sample_and_decode(
        args, flow_model, decoder, cond_codes, num_samples, device
    )

    import pickle
    with open(graph_save_path + 'generated_graphs.pkl', 'wb') as f:
        pickle.dump(generated_graphs, f)
    torch.save(samples_klein, graph_save_path + 'generated_klein_samples.pt')
    torch.save(sampled_cond, graph_save_path + 'generated_cond_samples.pt')

    results = evaluate_and_save_results(
        args, generated_graphs, train_list_adj, test_list_adj, graph_save_path
    )

    print("\n" + "=" * 60)
    print("Klein GraphTask Complete!")
    print("=" * 60)
    if "avg_mmd" in results:
        print(f"Final Average MMD: {results['avg_mmd']:.6f}")
    else:
        print(
            f"Final V.U.N.: {results['vun']:.6f}; "
            f"Average Ratio: {results['average_ratio']:.6f}"
        )

    torch.cuda.empty_cache()
    return results
