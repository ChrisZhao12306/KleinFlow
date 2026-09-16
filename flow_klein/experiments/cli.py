"""Dataset-aware entry point for random and anchor-based hyperparameter search."""
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

from flow_klein.paths import search_directory
from flow_klein.registry import dataset_spec


def run_random_search(dataset, argv):
    spec = dataset_spec(dataset)
    module = import_module('flow_klein.experiments.search_' + spec.name)
    parser = argparse.ArgumentParser(description='KleinFlow hyperparameter search for {}'.format(spec.name))
    parser.add_argument('--device', default=module.DEVICE)
    parser.add_argument('--num-experiments', type=int, default=module.NUM_EXPERIMENTS)
    parser.add_argument('--epoch-number', type=int, default=None)
    parser.add_argument('--epoch-diff', type=int, default=None)
    parser.add_argument('--training-seed', type=int, default=None)
    parser.add_argument('--search-seed', type=int, default=42)
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--results-file', default=None)
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args(argv)
    if options.num_experiments <= 0:
        parser.error('--num-experiments must be positive')
    module.DEVICE = options.device
    module.NUM_EXPERIMENTS = options.num_experiments
    module.SEARCH_SEED = options.search_seed
    for field in ('epoch_number', 'epoch_diff', 'training_seed'):
        value = getattr(options, field)
        if value is not None:
            module.FIXED_HYPERPARAMS['seed' if field == 'training_seed' else field] = value
    if options.dry_run:
        configs = module.generate_random_configs(options.num_experiments, seed=options.search_seed)
        print(json.dumps(dict(dataset=spec.name, pipeline=spec.pipeline, device=options.device,
                              results_name=module.RESULTS_FILE,
                              configurations=[{**module.FIXED_HYPERPARAMS, **c} for c in configs]), indent=2))
        return
    module.RUN_DIRECTORY = (Path(options.log_dir).resolve() if options.log_dir else search_directory(spec.name))
    module.RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    module.RESULTS_FILE = str(Path(options.results_file).resolve() if options.results_file else
                              module.RUN_DIRECTORY / module.RESULTS_FILE)
    Path(module.RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    module.main()


def main(argv=None, dataset=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if dataset is None:
        selector = argparse.ArgumentParser(add_help=False)
        selector.add_argument('--dataset', required=True)
        selected, argv = selector.parse_known_args(argv)
        dataset = selected.dataset
    try:
        spec = dataset_spec(dataset)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if spec.name in {'planar', 'tree'}:
        from flow_klein.experiments.structural import run_search
        return run_search(spec.name, argv)
    if spec.name in {'ego_small', 'community_small'}:
        from flow_klein.experiments.standard import run_search
        return run_search(spec.name, argv)
    return run_random_search(spec.name, argv)
