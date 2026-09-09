"""Opt-in comparison with source checkouts, in isolated interpreters on Linux."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flow_klein.paths import ROOT, ORCA_ROOT
from flow_klein.registry import DATASETS


@pytest.mark.parametrize('name', list(DATASETS))
def test_reference_forward_gradients_sampling_and_metrics(name, tmp_path):
    spec = DATASETS[name]
    variable = 'FLOW_KLEIN_SOURCE_' + spec.pipeline.upper()
    original = os.environ.get(variable)
    if not original:
        pytest.skip('Set {} to opt into original-versus-merged numerical comparison'.format(variable))
    pytest.importorskip('torch')
    pytest.importorskip('dgl')
    traces = []
    for kind, repository in [('original', original), ('merged', str(ROOT))]:
        directory = tmp_path / kind
        directory.mkdir()
        output = directory / 'trace.npz'
        command = [sys.executable, '-B', str(Path(__file__).with_name('_numerical_worker.py')),
                   '--root', repository, '--kind', kind, '--pipeline', spec.pipeline,
                   '--dataset', spec.training_name, '--output', str(output),
                   '--orca-dir', str(ORCA_ROOT),
                   '--device', os.environ.get('FLOW_KLEIN_TEST_DEVICE', 'cpu')]
        with (directory / 'worker.log').open('w') as log:
            completed = subprocess.run(command, cwd=str(directory), stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=300)
        assert completed.returncode == 0, (directory / 'worker.log').read_text()
        with np.load(output) as archive:
            traces.append({key: archive[key] for key in archive.files})
    assert traces[0].keys() == traces[1].keys()
    for key, reference in traces[0].items():
        actual = traces[1][key]
        if key.startswith(('train/', 'validation/', 'test/', 'decoded/', 'mask')):
            np.testing.assert_array_equal(reference, actual, err_msg=key)
        else:
            np.testing.assert_allclose(reference, actual, rtol=1e-5, atol=1e-6, equal_nan=False, err_msg=key)
