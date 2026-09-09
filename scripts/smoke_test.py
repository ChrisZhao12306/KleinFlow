"""Run short end-to-end training checks on the target Linux/CUDA machine."""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from . import _bootstrap
except ImportError:
    import _bootstrap

from flow_klein.registry import DATASETS, dataset_spec
from flow_klein.paths import ROOT, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    specs = [dataset_spec(name) for name in options.datasets]
    commands = []
    for spec in specs:
        commands.append([sys.executable, '-u', str(ROOT/'scripts/train.py'), '--dataset', spec.name,
                         '--device', options.device, '--UseGPU', str(options.device != 'cpu'),
                         '--epoch_number', '1', '--epoch_diff', '1', '--flow_steps', '2',
                         '--graphEmDim', '8', '--cond_dim', '8', '--decoder_node_dim', '16',
                         '--encoder_blocks', '1', '--lap_pe_dim', '2', '--batchSize', '8',
                         '--dit_hidden_dim', '32', '--dit_num_heads', '4', '--dit_num_layers', '1'])
    if options.dry_run:
        print(json.dumps(commands, indent=2))
        return
    if platform.system() != 'Linux':
        parser.error('End-to-end smoke tests target the Linux Klein_FM environment')
    import torch
    if options.device != 'cpu' and not torch.cuda.is_available():
        parser.error('Requested CUDA device is unavailable')
    parent = OUTPUT_ROOT / 'smoke'
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='run_', dir=str(parent)))
    results = []
    for spec, command in zip(specs, commands):
        destination = directory / spec.name
        destination.mkdir()
        command += ['--graph_save_path', str(destination)]
        with (destination/'console.log').open('w') as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=options.timeout)
                status = 'passed' if completed.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                status = 'timeout'
        if status == 'passed':
            required = ['metrics.json', 'generated_graphs.pkl', 'klein_encoder_final.pt', 'klein_flow_model_state.pt']
            if not all((destination/name).is_file() for name in required):
                status = 'missing_outputs'
        results.append(dict(dataset=spec.name, pipeline=spec.pipeline, status=status, directory=str(destination)))
        (directory/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print('{}: {}'.format(spec.name, status), flush=True)
    print('Results: {}'.format(directory))
    if any(row['status'] != 'passed' for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
