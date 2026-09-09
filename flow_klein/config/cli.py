"""Select the original argument parser before parsing model parameters."""
import argparse
from importlib import import_module

from flow_klein.registry import dataset_spec


def build_parser(dataset):
    spec = dataset_spec(dataset)
    module = import_module('flow_klein.config.' + spec.pipeline)
    parser = module.parser
    parser.description = 'Klein Flow Matching: {} pipeline'.format(spec.pipeline)
    for action in parser._actions:
        if action.dest == 'taskselect':
            action.choices = ['klein_graphtask']
    return parser


def parse_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--dataset', default='MUTAG')
    selected, _ = selector.parse_known_args(argv)
    try:
        spec = dataset_spec(selected.dataset)
    except ValueError as exc:
        selector.error(str(exc))
    parser = build_parser(spec.name)
    args = parser.parse_args(argv)
    args.dataset = spec.training_name
    if spec.pipeline == 'v0901':
        from flow_klein.config.v0901 import apply_klein_dataset_defaults
        apply_klein_dataset_defaults(args)
    return args
