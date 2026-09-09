"""Machine-readable results shared by both training implementations."""
import json
import os
import platform
from pathlib import Path

from flow_klein.registry import dataset_spec


def write_metrics_json(args, results, graph_save_path, timestamp, outer_seed, inner_validation):
    import torch

    spec = dataset_spec(args.dataset)
    benchmark = spec.name in {'planar', 'tree', 'ego_small', 'community_small'}
    gpu_name = None
    if torch.cuda.is_available() and getattr(args, 'UseGPU', True):
        try:
            index = torch.device(args.device).index
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device() if index is None else index)
        except (AssertionError, RuntimeError, ValueError):
            pass

    def safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        return safe(value.item()) if hasattr(value, 'item') else str(value)

    payload = dict(dataset=args.dataset, pipeline=spec.pipeline,
                   training_seed=int(getattr(args, 'seed', 0)),
                   outer_split_seed=None if benchmark else outer_seed,
                   validation_split_seed=(int(getattr(args, 'split_seed', 1432))
                                          if inner_validation and not benchmark else None),
                   split_strategy=('upstream_fixed' if benchmark else
                                   'seeded_80_20_plus_inner_val' if inner_validation else 'seeded_80_20'),
                   metric_profile=spec.metrics, timestamp=timestamp, device=str(args.device),
                   gpu_name=gpu_name, python_version=platform.python_version(),
                   python_implementation=platform.python_implementation(),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   hyperparameters={key: safe(value) for key, value in vars(args).items()
                                    if not key.startswith('_')},
                   metrics={key: safe(value) for key, value in results.items()})
    payload.update(payload['metrics'])
    path = Path(graph_save_path) / 'metrics.json'
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    os.replace(str(temporary), str(path))
