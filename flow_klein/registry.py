"""The dataset-to-implementation mapping is fixed for reproducible experiments."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    training_name: str
    pipeline: str
    metrics: str


DATASETS = {
    'planar': DatasetSpec('planar', 'planar', 'v0901', 'vun_ratio'),
    'tree': DatasetSpec('tree', 'tree', 'v0901', 'vun_ratio'),
    'grid': DatasetSpec('grid', 'grid', 'v0901', 'degree_clustering_spectral'),
    'ego': DatasetSpec('ego', 'SynEgo1000_original', 'fixed', 'degree_clustering_spectral'),
    'community': DatasetSpec('community', 'SynCommunity1000_origin', 'fixed', 'degree_clustering_spectral'),
    'ego_small': DatasetSpec('ego_small', 'ego_small', 'fixed', 'degree_clustering_orbit'),
    'community_small': DatasetSpec('community_small', 'community_small', 'fixed', 'degree_clustering_orbit'),
    'imdbbinary': DatasetSpec('imdbbinary', 'IMDBBINARY', 'fixed', 'degree_clustering_spectral'),
    'mutag': DatasetSpec('mutag', 'MUTAG', 'fixed', 'degree_clustering_spectral'),
}
ALIASES = {
    'synego1000_origin': 'ego', 'synego1000_original': 'ego', 'synego1000': 'ego',
    'syncommunity1000_origin': 'community', 'syncommunity1000': 'community',
    'comm20': 'community_small', 'communtiy': 'community',
}


def dataset_spec(name):
    key = str(name).strip().lower().replace('-', '_')
    key = ALIASES.get(key, key)
    if key not in DATASETS:
        raise ValueError('Unknown dataset {!r}. Supported: {}'.format(name, ', '.join(DATASETS)))
    return DATASETS[key]


def training_module(name):
    return import_module('flow_klein.training.' + dataset_spec(name).pipeline)


def train(args):
    spec = dataset_spec(args.dataset)
    args.dataset = spec.training_name
    return training_module(args.dataset).klein_graphtask(args)
