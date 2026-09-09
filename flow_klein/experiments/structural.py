"""Anchor-based local and global hyperparameter search for Planar and Tree."""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from flow_klein.paths import ROOT, OUTPUT_ROOT
from flow_klein.experiments.anchors import SEARCH_ANCHORS
from copy import deepcopy
from typing import Dict, List, Mapping, Tuple


TASK = "klein_graphtask"
SEARCH_METHOD = "anchor_local_global"

# Dataset-specific search ranges and constant training settings.
GRID_SEARCH_SPACE = {
    "lap_pe_dim": [12, 16, 20],
    "encoder_blocks": [4, 5, 6],
    "graphEmDim": [56, 64, 80, 96],
    "lr": [1.5e-4, 2e-4, 2.5e-4],
    "batchSize": [12, 16, 20],
    "decoder_node_dim": [160, 192, 224],
    "dropout": [0.05, 0.10],
    "lr_diff": [2e-5, 3e-5, 5e-5],
    "flow_steps": [160, 200, 240],
    "flow_base_std": [0.8, 1.0],
    "dit_hidden_dim": [320, 384, 448],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [5, 6, 7],
    "degree_reg_weight": [0.15, 0.20, 0.30],
    "degree_aux_weight": [0.05, 0.10, 0.15],
}

# Dataset-specific search ranges and constant training settings.
PLANAR_COMPACT_SPACE = {
    "lap_pe_dim": [16, 20],
    "encoder_blocks": [4, 5, 6],
    "graphEmDim": [48, 56, 64],
    "lr": [1.5e-4, 2e-4, 2.5e-4],
    "batchSize": [12, 16, 20],
    "decoder_node_dim": [128, 160, 192],
    "dropout": [0.05, 0.10],
    "lr_diff": [2e-5, 3e-5, 5e-5],
    "flow_steps": [160, 200, 240],
    "flow_base_std": [0.8, 1.0],
    "dit_hidden_dim": [320, 384, 448],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [6, 7],
    "degree_reg_weight": [0.15, 0.20, 0.30],
    "degree_aux_weight": [0.05, 0.10, 0.15],
    "degree_profile_blend": [0.25, 0.50, 0.75],
    "constraint_noise_scale": [0.0, 0.05, 0.10],
}

# Dataset-specific search ranges and constant training settings.
TREE_COMPACT_SPACE = {
    "lap_pe_dim": [12, 16, 20],
    "encoder_blocks": [4, 5, 6],
    "graphEmDim": [80, 96, 112],
    "lr": [1.5e-4, 2e-4, 2.5e-4],
    "batchSize": [12, 16, 20],
    "decoder_node_dim": [128, 160, 192],
    "dropout": [0.05, 0.10],
    "lr_diff": [2e-5, 3e-5, 5e-5],
    "flow_steps": [160, 200, 240],
    "flow_base_std": [0.8, 1.0],
    "dit_hidden_dim": [256, 320, 384],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [4, 5, 6],
    "degree_reg_weight": [0.15, 0.20, 0.30],
    "degree_aux_weight": [0.05, 0.10, 0.15],
    "degree_profile_blend": [0.50, 0.75, 1.0],
    "constraint_noise_scale": [0.0, 0.05, 0.10],
}

# Dataset-specific search ranges and constant training settings.
PLANAR_WIDE_SPACE = {
    "lap_pe_dim": [12, 16, 20],
    "encoder_blocks": [4, 5, 6],
    "graphEmDim": [48, 56, 64, 80],
    "lr": [1.5e-4, 2e-4, 2.5e-4, 3e-4],
    "batchSize": [12, 16, 20, 24],
    "decoder_node_dim": [128, 160, 192],
    "dropout": [0.05, 0.10, 0.15],
    "lr_diff": [1e-5, 2e-5, 3e-5, 5e-5],
    "flow_steps": [120, 160, 200],
    "flow_base_std": [0.6, 0.8, 1.0],
    "dit_hidden_dim": [320, 384, 448],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [6, 7],
    "degree_reg_weight": [0.20, 0.30, 0.40],
    "degree_aux_weight": [0.00, 0.05, 0.10, 0.15],
    "degree_profile_blend": [0.00, 0.25, 0.50, 0.75],
    "constraint_noise_scale": [0.00, 0.05, 0.10, 0.15],
}

# Dataset-specific search ranges and constant training settings.
TREE_WIDE_SPACE = {
    "lap_pe_dim": [8, 12, 16, 20],
    "encoder_blocks": [4, 5, 6],
    "graphEmDim": [64, 80, 96, 112],
    "lr": [2e-4, 2.5e-4, 3e-4],
    "batchSize": [12, 16],
    "decoder_node_dim": [128, 160, 192],
    "dropout": [0.05, 0.10],
    "lr_diff": [2e-5, 3e-5, 4e-5, 5e-5],
    "flow_steps": [160, 200, 240],
    "flow_base_std": [0.8, 1.0, 1.2],
    "dit_hidden_dim": [256, 320, 384],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [3, 4, 5, 6],
    "degree_reg_weight": [0.05, 0.10, 0.15, 0.20],
    "degree_aux_weight": [0.05, 0.10, 0.15],
    "degree_profile_blend": [0.25, 0.50, 0.75],
    "constraint_noise_scale": [0.00, 0.05, 0.10, 0.15],
}

# Dataset-specific search ranges and constant training settings.
PLANAR_SEARCH_SPACE = {
    "lap_pe_dim": [16, 20],
    "encoder_blocks": [4, 5, 6],
    "graphEmDim": [48, 56, 64],
    "lr": [2e-4, 2.5e-4, 3e-4, 3.5e-4],
    "batchSize": [12, 16, 20, 24],
    "decoder_node_dim": [128, 160, 192],
    "dropout": [0.05, 0.10, 0.15],
    "lr_diff": [2e-5, 3e-5, 4e-5, 5e-5],
    "flow_steps": [100, 120, 160, 200],
    "flow_base_std": [0.8, 1.0, 1.2],
    "dit_hidden_dim": [288, 320, 384, 448],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [6, 7, 8],
    "degree_reg_weight": [0.20, 0.25, 0.30, 0.35, 0.40],
    "degree_aux_weight": [0.00, 0.025, 0.05, 0.10, 0.15],
    "degree_profile_blend": [0.00, 0.25, 0.50, 0.625, 0.75],
    "constraint_noise_scale": [0.00, 0.025, 0.05, 0.075, 0.10, 0.15],
}

# Dataset-specific search ranges and constant training settings.
TREE_SEARCH_SPACE = {
    "lap_pe_dim": [12, 16, 20, 24],
    "encoder_blocks": [5, 6, 7],
    "graphEmDim": [56, 64, 72, 80, 96],
    "lr": [2.5e-4, 3e-4, 3.5e-4, 4e-4],
    "batchSize": [12, 16, 20],
    "decoder_node_dim": [144, 160, 176],
    "dropout": [0.025, 0.05, 0.075],
    "lr_diff": [2e-5, 3e-5, 4e-5, 5e-5],
    "flow_steps": [120, 160, 200],
    "flow_base_std": [0.8, 0.9, 1.0],
    "dit_hidden_dim": [288, 320, 352],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [4, 5, 6],
    "degree_reg_weight": [0.125, 0.15, 0.175],
    "degree_aux_weight": [0.025, 0.05, 0.075, 0.10],
    "degree_profile_blend": [0.50, 0.625, 0.75, 0.875],
    "constraint_noise_scale": [0.025, 0.05, 0.075, 0.10],
}

# Dataset-specific search ranges and constant training settings.


@dataclass(frozen=True)
class SearchSpec:
    dataset: str
    grid: Mapping[str, List]
    guidance_scale: float
    search_method: str
    default_devices: Tuple[str, ...] = ("cuda:0",)
    default_experiments: int = 60

    @property
    def results_file(self):
        return f"{self.dataset}_{self.search_method}.txt"

    @property
    def log_stem(self):
        return f"{self.dataset}_hypsearch_{self.search_method.lower()}"


SEARCH_SPECS = {
    "planar": SearchSpec(
        "planar", PLANAR_SEARCH_SPACE, 1.15, SEARCH_METHOD,
        ("cuda:4", "cuda:5", "cuda:6", "cuda:7"), default_experiments=120
    ),
    "tree": SearchSpec(
        "tree", TREE_SEARCH_SPACE, 1.15, SEARCH_METHOD,
        ("cuda:1", "cuda:2", "cuda:3"), default_experiments=120
    )
}


def search_spec(dataset: str) -> SearchSpec:
    return SEARCH_SPECS[str(dataset).strip().lower()]


def baseline_snapshot(dataset: str) -> Dict:
    """Load immutable anchor configurations for local and global search."""
    return deepcopy(SEARCH_ANCHORS[dataset])


def generate_random_configs(
    grid: Mapping[str, List], num_configs: int, seed: int = 42
) -> List[Dict]:
    if num_configs < 0:
        raise ValueError("num_configs must be non-negative")
    rng = random.Random(seed)
    maximum_combinations = math.prod(len(values) for values in grid.values())
    if num_configs > maximum_combinations:
        raise ValueError(
            f"Requested {num_configs} configs from only {maximum_combinations} combinations"
        )

    configurations = []
    seen = set()
    attempts = 0
    while len(configurations) < num_configs:
        attempts += 1
        if attempts > max(10000, num_configs * 1000):
            raise ValueError("Cannot sample enough unique attention-compatible configurations")
        config = {parameter: rng.choice(values) for parameter, values in grid.items()}
        if config["dit_hidden_dim"] % config["dit_num_heads"] != 0:
            continue
        signature = tuple(sorted(config.items()))
        if signature in seen:
            continue
        seen.add(signature)
        configurations.append(config)
    return configurations


def generate_experiments(spec: SearchSpec, options) -> List[Dict]:
    """Allocate 3 anchor, 93 local, and 24 global trials for a budget of 120.

    Each anchor has 31 variants: 11 single-, 10 double-, 10 triple-parameter
    changes. Other budgets use up to three anchors, then floor(0.8 * remaining)
    local trials (e.g. 3 + 45 + 12 trials at N=60).
    """
    count = options.num_experiments
    if count <= 0:
        raise ValueError("--num-experiments must be positive")
    fixed = fixed_hyperparameters(spec, options)
    if spec.dataset not in {"planar", "tree"}:
        return [dict(exp_id=i, kind="random", config={**fixed, **config})
                for i, config in enumerate(generate_random_configs(spec.grid, count, options.search_seed))]
    anchors = baseline_snapshot(spec.dataset)["anchors"]
    rng = random.Random(options.search_seed)
    rows, seen = [], set()

    def add(config, kind, source=None, changed=()):
        key = tuple((k, config[k]) for k in spec.grid)
        if key in seen or config["dit_hidden_dim"] % config["dit_num_heads"]:
            return False
        if any(config[k] not in values for k, values in spec.grid.items()):
            raise ValueError("Baseline contains a value outside the search space")
        seen.add(key)
        rows.append(dict(exp_id=len(rows), kind=kind, source_exp_id=source,
                         changed_parameters=sorted(changed), config=config))
        return True

    for anchor in anchors[:count]:
        config = {**anchor["config"], "epoch_number": options.epoch_number,
                  "epoch_diff": options.epoch_diff, "seed": options.training_seed}
        if not add(config, "baseline", anchor["source_exp_id"]):
            raise ValueError("Baselines must be unique and attention-compatible")
    local_count = (max(0, count - len(rows)) * 4) // 5
    for local_index in range(local_count):
        anchor = anchors[local_index % len(anchors)]
        mutation_count = 1 + (local_index // len(anchors)) % 3
        for attempt in range(10000):
            config = {**fixed, **anchor["config"], "epoch_number": options.epoch_number,
                      "epoch_diff": options.epoch_diff, "seed": options.training_seed}
            changed = rng.sample(list(spec.grid), mutation_count)
            for key in changed:
                values = sorted(spec.grid[key])
                position = values.index(config[key])
                neighbors = values[max(0, position - 1):position] + values[position + 1:position + 2]
                config[key] = rng.choice(neighbors)
            if add(config, "local", anchor["source_exp_id"], changed):
                break
        else:
            raise ValueError("Local search space exhausted; reduce --num-experiments")
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > max(10000, count * 1000):
            raise ValueError("Global search space exhausted")
        config = {**fixed, **{k: rng.choice(values) for k, values in spec.grid.items()}}
        add(config, "random")
    return rows


def fixed_hyperparameters(spec: SearchSpec, options) -> Dict:
    if spec.dataset in {"planar", "tree"}:
        fixed = dict(baseline_snapshot(spec.dataset)["anchors"][0]["config"])
    else:
        from flow_klein.config.recipes import DATASET_DEFAULTS
        fixed = dict(DATASET_DEFAULTS[spec.dataset])
    fixed.update(
        {
            "epoch_number": options.epoch_number,
            "epoch_diff": options.epoch_diff,
            "flow_integrator": "heun",
            "directed": False,
            "bfsOrdering": False,
            "node_feat_mode": "struct",
            "flow_cond_dropout": 0.1,
            "use_cond_guidance": True,
            "flow_guidance_scale": spec.guidance_scale,
            "seed": options.training_seed,
        }
    )
    return fixed


def _training_args(spec: SearchSpec, options, config: Dict, experiment_id: int):
    from flow_klein.config.structural import parser as training_parser

    fixed = fixed_hyperparameters(spec, options)
    argument_list = [
        "--taskselect", TASK,
        "--dataset", spec.dataset,
        "--device", options.device,
        "--UseGPU", str(options.device.lower() != "cpu"),
    ]
    for parameter, value in {**fixed, **config}.items():
        argument_list.extend([f"--{parameter}", str(value)])
    save_path = Path(options.graph_save_path)
    argument_list.extend(["--graph_save_path", str(save_path) + "/"])
    args = training_parser.parse_args(argument_list)

    # Keep explicit boolean values intact for older server parser versions too.
    for parameter, value in {**fixed, **config}.items():
        if isinstance(value, bool):
            setattr(args, parameter, value)
    args.UseGPU = options.device.lower() != "cpu"
    return args, fixed


def run_experiment(spec: SearchSpec, options, config: Dict, experiment_id: int) -> Dict:
    full_config = {**fixed_hyperparameters(spec, options), **config}
    print("\n" + "=" * 72)
    print(f"Experiment {experiment_id + 1}/{options.num_experiments}: {full_config}")
    print("=" * 72)
    started = time.time()
    try:
        import torch
        from flow_klein.training.structural import klein_graphtask

        args, _ = _training_args(spec, options, config, experiment_id)
        if options.device != "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable; CPU fallback is disabled")
            torch.cuda.set_device(options.device)
        metrics = klein_graphtask(args)
        metrics = validate_metrics(spec, metrics)
        return {
            "exp_id": experiment_id,
            "config": full_config,
            "metrics": {key: float(value) for key, value in metrics.items()},
            "elapsed_time": time.time() - started,
            "status": "success",
        }
    except Exception as exc:
        traceback.print_exc()
        return {
            "exp_id": experiment_id,
            "config": full_config,
            "metrics": {},
            "elapsed_time": time.time() - started,
            "status": f"failed: {exc}",
        }


def validate_metrics(spec, metrics):
    values = {key: float(value) for key, value in metrics.items()}
    required = ("vun", "average_ratio", "frac_valid") if spec.dataset in {"planar", "tree"} else ("vun", "average_ratio")
    missing = [key for key in required if key not in values]
    non_finite = [key for key, value in values.items() if not math.isfinite(value)]
    if missing or non_finite:
        raise ValueError(f"Invalid evaluation result; missing={missing}, non_finite={non_finite}")
    if spec.dataset in {"planar", "tree"} and values["frac_valid"] != 1.0:
        raise ValueError(f"Constrained benchmark must have frac_valid=1.0; got {values['frac_valid']}")
    return values


def result_sort_key(result: Dict):
    metrics = result["metrics"]
    vun = float(metrics.get("vun", float("-inf")))
    average_ratio = float(metrics.get("average_ratio", float("inf")))
    if not math.isfinite(vun):
        vun = float("-inf")
    if not math.isfinite(average_ratio) or average_ratio < 0:
        average_ratio = float("inf")
    return -vun, average_ratio


METRIC_COLUMNS = (
    "vun",
    "average_ratio",
    "degree_ratio",
    "clustering_ratio",
    "orbit_ratio",
    "spectre_ratio",
    "wavelet_ratio",
)


def reproduction_command(spec: SearchSpec, options, config: Mapping, result=None) -> str:
    result = result or {}
    device = result.get("logical_device") or options.device or spec.default_devices[0]
    arguments = [
        "python",
        str(ROOT / "scripts" / "train.py"),
        "--taskselect",
        TASK,
        "--dataset",
        spec.dataset,
        "--device",
        device,
    ]
    for parameter, value in config.items():
        arguments.extend([f"--{parameter}", str(value)])
    command = " ".join(shlex.quote(value) for value in arguments)
    visible = result.get("cuda_visible_devices")
    if visible is not None:
        command = "env " + shlex.quote("CUDA_VISIBLE_DEVICES=" + visible) + " " + command
    # Re-running a printed command also gets a fresh model directory. `&&`
    # prevents training if mktemp cannot reserve it. Commands target Bash.
    root = Path(getattr(options, "run_dir", ".")).resolve()
    template = str(root / f"{spec.dataset}_{spec.search_method}_reproduce_XXXXXX")
    return (f"reproduce_dir=$(mktemp -d {shlex.quote(template)}) && " + command
            + ' --graph_save_path "$reproduce_dir/"')


def save_results(spec: SearchSpec, options, results: List[Dict]) -> None:
    from flow_klein.experiments.runtime import atomic_text_writer, reserve_results

    successful = [result for result in results if result["status"] == "success"]
    failed = [result for result in results if result["status"] != "success"]
    successful.sort(key=result_sort_key)
    destination = Path(options.results_file)
    reserve_results(options)

    with atomic_text_writer(destination) as handle:
        handle.write("=" * 130 + "\n")
        handle.write("KLEINFLOW HYPERPARAMETER SEARCH\n")
        handle.write("=" * 130 + "\n")
        handle.write(f"Dataset: {spec.dataset}\n")
        handle.write(f"Search method: {spec.search_method}\n")
        handle.write(f"Run directory: {getattr(options, 'run_dir', 'N/A')}\n")
        handle.write(f"Run status: {getattr(options, 'run_status', 'running')}\n")
        handle.write("Metric profile: vun_ratio\n")
        handle.write(f"Physical devices: {getattr(options, 'devices', None) or [options.device]}\n")
        handle.write(f"Completed: {len(results)}/{options.num_experiments}\n")
        handle.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.write("SEARCH SPACE\n")
        for parameter, values in spec.grid.items():
            handle.write(f"  {parameter}: {values}\n")

        handle.write("\nRESULTS (V.U.N. descending, Average Ratio ascending)\n")
        header = f"{'Rank':<6}{'Exp':<6}"
        for column in METRIC_COLUMNS:
            header += f"{column[:17]:<19}"
        header += f"{'Time(s)':<12}Status\n"
        handle.write(header)
        handle.write("-" * 170 + "\n")
        for rank, result in enumerate(successful, 1):
            line = f"{rank:<6}{result['exp_id']:<6}"
            for column in METRIC_COLUMNS:
                if column in result["metrics"]:
                    line += f"{result['metrics'][column]:<19.6f}"
                else:
                    line += f"{'N/A':<19}"
            line += f"{result['elapsed_time']:<12.1f}{result['status']}\n"
            handle.write(line)
        for result in failed:
            handle.write(
                f"{'--':<6}{result['exp_id']:<6}"
                + f"{'NaN':<19}" * len(METRIC_COLUMNS)
                + f"{result['elapsed_time']:<12.1f}{result['status']}\n"
            )
        handle.write(
            "\nN/A means the train/test reference distance rounds to 0.0000; "
            "that ratio is undefined and is excluded from average_ratio.\n"
        )

        handle.write("\nDETAILED CONFIGURATIONS\n")
        handle.write("=" * 130 + "\n")
        for rank, result in enumerate(successful, 1):
            handle.write(f"Rank {rank} (Exp {result['exp_id']}):\n")
            handle.write(f"  Sampling: {result.get('kind', 'random')}; source Exp: {result.get('source_exp_id', 'N/A')}\n")
            handle.write(f"  Physical device: {result.get('physical_device', options.device)}; logical: {result.get('logical_device', options.device)}\n")
            handle.write(f"  Artifacts: {result.get('experiment_dir', 'N/A')}\n")
            handle.write(f"  Metrics: {result['metrics']}\n")
            for parameter, value in result["config"].items():
                handle.write(f"  --{parameter} {value}\n")
            handle.write(
                f"  Reproduce: {reproduction_command(spec, options, result['config'], result)}\n"
            )
            handle.write("\n")
        for result in failed:
            handle.write(
                f"Failed Exp {result['exp_id']}: {result['status']}\n"
                f"  Config: {result['config']}\n"
                f"  Reproduce: "
                f"{reproduction_command(spec, options, result['config'], result)}\n\n"
            )

        if successful:
            best = successful[0]
            handle.write("BEST CONFIGURATION COMMAND\n")
            handle.write(reproduction_command(spec, options, best["config"], best) + "\n")


def build_parser(spec: SearchSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"KleinFlow hyperparameter search for {spec.dataset}"
    )
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", help="One physical CUDA device, or cpu")
    devices.add_argument("--devices", nargs="+", help="Physical devices, e.g. cuda:1 cuda:2 cuda:3")
    parser.add_argument("--num-experiments", type=int, default=spec.default_experiments)
    parser.add_argument("--epoch-number", type=int, default=2000)
    parser.add_argument("--epoch-diff", type=int, default=2000)
    parser.add_argument("--results-file", help="Optional NEW result file; existing paths are never overwritten")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=1432)
    parser.add_argument("--log-dir", default=str(OUTPUT_ROOT / "search"), help="Parent directory for unique run folders")
    parser.add_argument("--background", action="store_true", help="Detach the controller and print its log/PID paths")
    parser.add_argument("--check-only", action="store_true", help="Check data, metrics, training imports and every GPU, without training")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download and validate the dataset, then exit without training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete experiment plan without training dependencies or output writes",
    )
    return parser


def run_search(dataset: str, argv=None) -> None:
    from flow_klein.experiments.runtime import launch_search

    spec = search_spec(dataset)
    launch_search(spec, build_parser(spec).parse_args(argv))
