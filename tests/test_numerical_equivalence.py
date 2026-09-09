"""Compare deterministic model traces with externally supplied reference arrays."""
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
    reference_dir = os.environ.get('FLOW_KLEIN_REFERENCE_DIR')
    if not reference_dir:
        pytest.skip('Set FLOW_KLEIN_REFERENCE_DIR to enable numerical reference comparisons')
    pytest.importorskip('torch')
    pytest.importorskip('dgl')
    reference_path = Path(reference_dir) / (name + '.npz')
    assert reference_path.is_file(), 'Missing reference trace: {}'.format(reference_path)
    output = tmp_path / 'trace.npz'
    command = [sys.executable, '-B', str(Path(__file__).with_name('_numerical_worker.py')),
               '--root', str(ROOT), '--pipeline', spec.pipeline,
               '--dataset', spec.training_name, '--output', str(output),
               '--orca-dir', str(ORCA_ROOT),
               '--device', os.environ.get('FLOW_KLEIN_TEST_DEVICE', 'cpu')]
    with (tmp_path / 'worker.log').open('w') as log:
        completed = subprocess.run(command, cwd=str(tmp_path), stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=300)
    assert completed.returncode == 0, (tmp_path / 'worker.log').read_text()
    with np.load(reference_path, allow_pickle=False) as archive:
        references = {key: archive[key] for key in archive.files}
    with np.load(output, allow_pickle=False) as archive:
        actuals = {key: archive[key] for key in archive.files}
    assert references.keys() == actuals.keys()
    for key, reference in references.items():
        actual = actuals[key]
        if key.startswith(('train/', 'validation/', 'test/', 'decoded/', 'mask')):
            np.testing.assert_array_equal(reference, actual, err_msg=key)
        else:
            np.testing.assert_allclose(reference, actual, rtol=1e-5, atol=1e-6, equal_nan=False, err_msg=key)
