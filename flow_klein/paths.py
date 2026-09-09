"""Repository-relative defaults; explicit output paths remain caller-relative."""
import os
from datetime import datetime
from pathlib import Path

from flow_klein.registry import dataset_spec

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('FLOW_KLEIN_DATA_DIR', str(ROOT / 'data'))).resolve()
OUTPUT_ROOT = Path(os.environ.get('FLOW_KLEIN_OUTPUT_DIR', str(ROOT / 'outputs'))).resolve()
ORCA_ROOT = ROOT / 'third_party' / 'orca'


def prepare_training_args(args, expected_pipeline):
    spec = dataset_spec(args.dataset)
    if spec.pipeline != expected_pipeline:
        raise ValueError('{} must use {}, not {}'.format(spec.name, spec.pipeline, expected_pipeline))
    if getattr(args, 'taskselect', 'klein_graphtask') != 'klein_graphtask':
        raise ValueError('Only --taskselect klein_graphtask is supported')
    args.dataset = spec.training_name
    path = getattr(args, 'graph_save_path', None)
    if path is None:
        run = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + str(os.getpid())
        path = OUTPUT_ROOT / spec.name / run
    args.graph_save_path = str(Path(path).expanduser().resolve()) + os.sep
    args.pipeline = expected_pipeline


def search_directory(dataset):
    """Reserve a new directory without consuming training RNG state."""
    import tempfile
    parent = OUTPUT_ROOT / 'search' / dataset_spec(dataset).name
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=str(parent)))
