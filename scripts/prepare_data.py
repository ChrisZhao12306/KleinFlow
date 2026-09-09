"""Prepare the nine supported datasets without running training."""
import argparse
import json

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap

from flow_klein.registry import DATASETS, dataset_spec
from flow_klein.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', default=list(DATASETS))
    parser.add_argument('--dry-run', action='store_true', help='Show sources without importing training dependencies')
    args = parser.parse_args()
    try:
        specs = [dataset_spec(name) for name in args.datasets]
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(data_dir=str(DATA_ROOT), datasets=[dict(name=s.name, pipeline=s.pipeline) for s in specs]), indent=2))
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if spec.name in {'planar', 'tree', 'ego_small', 'community_small'}:
            from importlib import import_module
            loader = import_module('flow_klein.data.benchmarks_' + spec.pipeline)
            splits = loader.load_benchmark_splits(spec.name)
            print('{}: {}'.format(spec.name, loader.split_sizes(splits)))
        else:
            from importlib import import_module
            loader = import_module('flow_klein.data.' + spec.pipeline)
            graphs, _, _ = loader.list_graph_loader(spec.training_name, shuffle=False)
            print('{}: {} graphs'.format(spec.name, len(graphs)))


if __name__ == '__main__':
    main()
