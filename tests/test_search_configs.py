import hashlib
import json
import sys
import types
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from flow_klein.config.cli import parse_args
from flow_klein.registry import dataset_spec
from flow_klein.experiments import standard, structural

CONTRACTS = json.loads((Path(__file__).parent / 'fixtures/reference_contracts.json').read_text())


@pytest.mark.parametrize('dataset', ['grid', 'ego', 'community', 'imdbbinary', 'mutag'])
def test_seeded_random_search_and_training_arguments(dataset, tmp_path, monkeypatch):
    module = import_module('flow_klein.experiments.search_' + dataset)
    for name, expected in CONTRACTS['search_constants'][dataset].items():
        assert getattr(module, name) == expected
    configs = module.generate_random_configs(module.NUM_EXPERIMENTS)
    encoded = json.dumps(configs, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(encoded).hexdigest() == CONTRACTS['search_sample_sha256'][dataset]
    seen = []

    def train(args):
        seen.append(args)
        return dict(mmd_degree=0.1, mmd_clustering=0.2, mmd_spectral=0.3, avg_mmd=0.2)

    fake = types.ModuleType('flow_klein.training.' + dataset_spec(dataset).pipeline)
    fake.klein_graphtask = train
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(module, 'RUN_DIRECTORY', tmp_path)
    for index, config in enumerate(configs):
        result = module.run_experiment(config, index)
        assert result['status'] == 'success'
        args = seen[-1]
        for key, value in {**module.FIXED_HYPERPARAMS, **config}.items():
            assert getattr(args, key) == value, key
        assert args.device == module.DEVICE
        assert args.dataset == module.DATASET
        assert Path(args.graph_save_path).parent == tmp_path


@pytest.mark.parametrize('dataset', ['planar', 'tree', 'ego_small', 'community_small'])
def test_benchmark_search_parameters_survive_parsing(dataset, tmp_path):
    module = structural if dataset in {'planar', 'tree'} else standard
    spec = module.search_spec(dataset)
    options = module.build_parser(spec).parse_args([])
    options.device = 'cpu'
    options.log_dir = str(tmp_path)
    options.graph_save_path = str(tmp_path / 'model')
    if module is structural:
        configs = [row['config'] for row in module.generate_experiments(spec, options)]
    else:
        configs = module.generate_random_configs(spec.grid, options.num_experiments, options.search_seed)
    for index, config in enumerate(configs):
        args, base = module._training_args(spec, options, config, index)
        for key, value in {**base, **config}.items():
            assert getattr(args, key) == value
        assert args.UseGPU is False
        assert args.dataset == dataset


@pytest.mark.parametrize('dataset', ['grid', 'ego', 'community', 'imdbbinary', 'mutag'])
def test_search_result_writers_and_commands(dataset, tmp_path):
    module = import_module('flow_klein.experiments.search_' + dataset)
    result = dict(exp_id=0, config={'seed':1432}, mmd_degree=0.1, mmd_clustering=0.2,
                  mmd_spectral=0.3, avg_mmd=0.2, elapsed_time=1, status='success')
    destination = tmp_path / 'results.txt'
    module.save_results([result], str(destination))
    output = destination.read_text()
    assert 'scripts' in output and 'train.py' in output
    assert 'main.py' not in output
