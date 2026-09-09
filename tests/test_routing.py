import hashlib
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from flow_klein.config.cli import build_parser, parse_args
from flow_klein.config.recipes import DATASET_DEFAULTS
from flow_klein.paths import ROOT, prepare_training_args
from flow_klein.registry import DATASETS, dataset_spec

CONTRACTS = json.loads((Path(__file__).parent / 'fixtures/reference_contracts.json').read_text())


@pytest.mark.parametrize('name', list(DATASETS))
def test_reference_defaults_and_dataset_recipes(name):
    spec = dataset_spec(name)
    parser = build_parser(name)
    reference = CONTRACTS['default_parameters'][spec.pipeline]
    for action in parser._actions:
        if action.dest not in {'help', 'taskselect'}:
            assert action.default == reference[action.dest], action.dest
    args = parse_args(['--dataset', name])
    assert args.dataset == spec.training_name
    if spec.pipeline == 'structural':
        for key, value in DATASET_DEFAULTS[spec.training_name].items():
            assert getattr(args, key) == value
    else:
        assert not hasattr(args, 'decode_mode')
        assert not hasattr(args, 'structure_constraint')
        assert not hasattr(args, 'val_ratio')


@pytest.mark.parametrize('alias, canonical', [
    ('PLANAR', 'planar'), ('Tree', 'tree'), ('Grid', 'grid'), ('Ego', 'ego'),
    ('SynEgo1000_origin', 'ego'), ('SynEgo1000_original', 'ego'),
    ('COMMUNITY', 'community'), ('SynCommunity1000_origin', 'community'),
    ('Ego-small', 'ego_small'), ('Community-small', 'community_small'),
    ('comm20', 'community_small'), ('imdbbinary', 'imdbbinary'), ('MUTAG', 'mutag')])
def test_aliases_select_one_pipeline(alias, canonical):
    assert dataset_spec(alias) == DATASETS[canonical]
    assert parse_args(['--dataset', alias]).dataset == DATASETS[canonical].training_name


def test_explicit_values_equal_to_parser_defaults_override_recipes():
    args = parse_args(['--dataset', 'planar', '--flow_steps', '200',
                       '--bfsOrdering', 'True', '--directed=False', '--candidate_multiplier', '3'])
    assert args.flow_steps == 200
    assert args.bfsOrdering is True
    assert args.directed is False
    assert args.candidate_multiplier == 3
    from flow_klein.config.structural import apply_klein_dataset_defaults
    apply_klein_dataset_defaults(args)
    assert args.candidate_multiplier == 3


@pytest.mark.parametrize('name', ['sbm', 'BA', 'PROTEINS', 'random_tree', 'missing'])
def test_out_of_scope_datasets_are_rejected(name):
    with pytest.raises(ValueError, match='Unknown dataset'):
        dataset_spec(name)


def test_wrong_internal_route_cannot_silently_train(tmp_path):
    args = parse_args(['--dataset', 'ego'])
    with pytest.raises(ValueError, match='must use standard'):
        prepare_training_args(args, 'structural')
    args.graph_save_path = str(tmp_path / 'output')
    prepare_training_args(args, 'standard')
    assert args.graph_save_path.endswith(os.sep)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('entry', ['train.py', 'prepare_data.py', 'hyperparam_search.py'])
def test_dry_runs_from_unrelated_directory_write_nothing(tmp_path, entry):
    flags = ['--dataset', 'tree'] if entry != 'prepare_data.py' else ['tree']
    env = {**os.environ, 'FLOW_KLEIN_OUTPUT_DIR': str(tmp_path / 'outputs'),
           'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / entry), *flags, '--dry-run'],
                            cwd=str(tmp_path), env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)
    assert list(tmp_path.iterdir()) == []


def test_copied_data_matches_source_hashes():
    for name, expected in CONTRACTS['data_sha256'].items():
        path = ROOT / 'data' / name
        if not path.exists():
            continue  # A code-only Git checkout may prepare/download data separately.
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_unique_environment_preserves_source_dependencies():
    environment = (ROOT / 'environment.yml').read_text()
    assert environment.startswith('name: Klein_FM\n')
    assert '\nprefix:' not in environment
    assert hashlib.sha256(environment.split('\n', 1)[1].encode()).hexdigest() == CONTRACTS['environment_body_sha256']


def test_search_anchors_match_reference_hashes():
    from flow_klein.experiments.anchors import SEARCH_ANCHORS
    encoded = json.dumps(SEARCH_ANCHORS, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == CONTRACTS['search_anchors_sha256']


def test_smoke_commands_cover_all_datasets_with_valid_arguments(tmp_path):
    completed = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/smoke_test.py'), '--dry-run'],
                               cwd=str(tmp_path), capture_output=True, text=True, check=True)
    commands = json.loads(completed.stdout)
    assert len(commands) == 9
    observed = set()
    for command in commands:
        args = parse_args(command[3:])
        observed.add(dataset_spec(args.dataset).name)
        assert args.epoch_number == args.epoch_diff == 1
        assert args.flow_steps == 2
    assert observed == set(DATASETS)
    assert list(tmp_path.iterdir()) == []
