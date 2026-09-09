"""Isolated old/new numerical trace; writes only to the requested test directory."""
import argparse
import importlib
import os
import random
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--kind', choices=['original', 'merged'], required=True)
parser.add_argument('--pipeline', choices=['fixed', 'v0901'], required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--orca-dir', required=True)
parser.add_argument('--device', default='cpu')
options = parser.parse_args()
sys.path.insert(0, options.root)

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

if options.kind == 'original':
    training = importlib.import_module('klein_graphtask')
    data = importlib.import_module('data')
    config = importlib.import_module('config')
    metrics = importlib.import_module('graph_evaluate.benchmark_metrics')
else:
    training = importlib.import_module('flow_klein.training.' + options.pipeline)
    data = importlib.import_module('flow_klein.data.' + options.pipeline)
    config = importlib.import_module('flow_klein.config.' + options.pipeline)
    metrics = importlib.import_module('flow_klein.evaluation.' + options.pipeline)
orca_dir = Path(options.orca_dir)
metrics._orca_paths = lambda: (orca_dir, orca_dir / 'orca')
random.seed(1432)
np.random.seed(1432)
torch.manual_seed(1432)
torch.cuda.manual_seed_all(1432)
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(options.device)
args = config.parser.parse_args(['--dataset', options.dataset, '--device', options.device,
                                 '--graphEmDim', '8', '--cond_dim', '8', '--encoder_blocks', '2',
                                 '--decoder_node_dim', '16', '--lap_pe_dim', '2', '--dropout', '0.1'])
if options.pipeline == 'v0901':
    config.apply_klein_dataset_defaults(args)

# Fixture injection avoids downloads and writes by the original data loaders.
# The original split, ordering, feature and padding functions still execute.
raw = [sp.csr_matrix(nx.to_scipy_sparse_array(
    nx.path_graph(4 + index % 4) if index % 2 else nx.cycle_graph(4 + index % 4),
    dtype=np.float32, format='csr')) for index in range(30)]
training.list_graph_loader = lambda *a, **kw: ([x.copy() for x in raw], [None] * len(raw), None)
training.load_benchmark_splits = lambda *a, **kw: {
    'train': [x.copy() for x in raw[:20]], 'val': [x.copy() for x in raw[20:25]],
    'test': [x.copy() for x in raw[25:]]}
loaded = training.load_data(args)
dataset, _, val_adj, test_adj, train_adj = loaded[:5]
dataset.processALL(self_for_none=True)
org_adj = dataset.adj_s[:2]
features = torch.cat(dataset.x_s[:2]).to(device)
batch_size = [2, dataset.max_num_nodes]
mask = training.build_node_mask(dataset.num_nodes[:2], batch_size[1], device)
graph = training.prepare_batch_graphs(org_adj, device)
profile = None
extra = {}
if options.pipeline == 'v0901':
    profile = training.DatasetProfile(n_bins=32).fit(train_adj)
    extra = dict(use_struct_cond=True, profile_stat_dim=profile.profile_stat_dim, struct_cond_dim=4)
encoder = training.KleinEncoder(in_feature_dim=dataset.feature_size, hidden_layers=[16, 16],
                                graph_latent_dim=8, cond_dim=8, encoder_blocks=2,
                                input_proj_dim=8, dropout=0.1, **extra)
full_cond_dim = encoder.full_cond_dim if options.pipeline == 'v0901' else 8
decoder = training.MaskedGraphDecoder(latent_dim=8, cond_dim=full_cond_dim,
                                     max_nodes=batch_size[1], directed=args.directed, node_dim=16)
model = training.KleinGraphVAE(encoder, decoder).to(device)
trace = {}


def record(prefix, value):
    if isinstance(value, dict):
        for key, item in value.items():
            record(prefix + '/' + key, item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            record(prefix + '/' + str(index), item)
    elif torch.is_tensor(value):
        trace[prefix] = value.detach().cpu().numpy().copy()
    elif sp.issparse(value):
        trace[prefix] = value.toarray()
    elif value is not None:
        trace[prefix] = np.asarray(value)


record('train', train_adj)
record('validation', val_adj)
record('test', test_adj)
record('features', features)
record('mask', mask)
record('initial_parameters', model.state_dict())
forward_kwargs = dict(node_mask=mask)
if profile is not None:
    forward_kwargs['profile_vec'] = training.compute_profile_vec_from_adj(org_adj, mask, profile, device)
    record('profile', forward_kwargs['profile_vec'])
output = model(graph, features, batch_size, **forward_kwargs)
record('forward', output)
adj_probs, samples, mean, log_std, cond, aux, adj_logits = output
_, target = training.get_subGraph_features(args, org_adj, None, None)
target = target.to(device)
stats = (torch.stack(dataset.graph_stats[:2]).to(device) if profile is not None
         else training.compute_graph_batch_stats(target, mask))
kwargs = dict(adj_logits=adj_logits, adj_probs=adj_probs, target_adj=target,
              node_logits=aux['node_logits'], node_mask=mask, stats_pred=aux['stats_pred'],
              stats_target=stats, encoder_aux=aux['encoder_aux'], log_std=log_std,
              mean=mean, kernel_model=None, target_kernel_val=None)
if profile is not None:
    kwargs['degree_pred'] = aux['degree_pred']
losses = training.compute_vae_loss_v2(**kwargs)
record('loss', losses)
losses['total'].backward()
record('gradients', {name: p.grad for name, p in model.named_parameters() if p.grad is not None})
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
optimizer.step()
record('updated_parameters', model.state_dict())

flow = training.KleinFlowMatching(dim=8, cond_dim=full_cond_dim, hidden_dim=32,
                                  num_heads=4, num_layers=1, num_timesteps=3,
                                  device=str(device)).to(device)
flow_loss = flow.loss_fn(samples.detach(), cond.detach())
record('flow_loss', flow_loss)
flow_loss.backward()
record('flow_gradients', {name: p.grad for name, p in flow.named_parameters() if p.grad is not None})
torch.optim.AdamW(flow.parameters(), lr=1e-4).step()
flow.eval()
record('euler', flow.sample(cond.detach(), steps=3))
record('heun', flow.sample_heun(cond.detach(), steps=3))

# Controlled decoder outputs exercise each dataset's original postprocessing.
n = batch_size[1]
edge_logits = torch.linspace(-2, 2, n*n, device=device).reshape(1, n, n).repeat(2, 1, 1)
node_logits = torch.ones(2, n, device=device)
statistics = torch.tensor([[n, n-1, 0, 0, 0, 0], [n, n-1, 0, 0, 0, 0]], device=device)
decode_args = dict(args=args, adj_probs=edge_logits.sigmoid(), node_logits=node_logits, stats_pred=statistics)
if profile is not None:
    decode_args.update(degree_pred=torch.ones(2, n, device=device)*2,
                       edge_logits_raw=edge_logits)
generated = training.decode_samples_to_graphs(**decode_args)
record('decoded', [nx.to_numpy_array(g, nodelist=list(g.nodes())) for g in generated])
reference = [nx.from_scipy_sparse_array(adj) for adj in test_adj]
if metrics.evaluation_profile(options.dataset) == 'degree_clustering_spectral':
    results = dict(degree=training.degree_stats(reference, generated),
                   clustering=training.clustering_stats(reference, generated),
                   spectral=training.spectral_stats(reference, generated))
else:
    results = metrics.evaluate_external_benchmark(options.dataset, generated,
               [nx.from_scipy_sparse_array(adj) for adj in train_adj], reference,
               cache_dir=Path(options.output).parent / 'metric_cache')
record('metrics', results)
np.savez_compressed(options.output, **trace)
# Each trace is an isolated CLI child. Avoid native-library shutdown hangs only
# after the archive has been fully written and closed.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
