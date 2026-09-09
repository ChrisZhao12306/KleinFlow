"""Random search for Ego-small and Community-small."""

from __future__ import annotations

import argparse
import math
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping


from flow_klein.registry import dataset_spec
from flow_klein.paths import ROOT, search_directory
import shlex
import json


def normalize_benchmark_name(name):
    spec = dataset_spec(name)
    if spec.name not in {"ego_small", "community_small"}:
        raise ValueError("Expected ego_small or community_small")
    return spec.name


def evaluation_profile(name):
    return dataset_spec(name).metrics



TASK = "klein_graphtask"

SMALL_GRAPH_GRID = {
    "graphEmDim": [40, 48, 56, 64],
    "lr": [1.5e-4, 2e-4, 2.5e-4, 3e-4, 4e-4],
    "batchSize": [24, 32, 40],
    "decoder_node_dim": [64, 96, 128],
    "dropout": [0.05, 0.1],
    "lr_diff": [1.5e-5, 2e-5, 3e-5, 5e-5],
    "flow_steps": [100, 120, 140, 160, 180],
    "flow_base_std": [0.8, 1.0, 1.2],
    "dit_hidden_dim": [224, 256, 288],
    "dit_num_heads": [4, 8],
    "dit_num_layers": [3, 4, 5],
}

# Dataset-specific search ranges and constant training settings.
COMMUNITY_SMALL_GRID = {
    **SMALL_GRAPH_GRID,
    "graphEmDim": [36, 40, 44, 48, 52],
    "lr": [1.75e-4, 2e-4, 2.25e-4, 2.5e-4, 3e-4],
    "batchSize": [40, 44, 48, 52],
    "dropout": [0.05, 0.075, 0.1],
    "lr_diff": [2e-5, 3e-5, 4e-5, 5e-5, 6e-5],
    "flow_steps": [100, 110, 120, 130, 140, 160],
    "flow_base_std": [0.8, 0.9, 1.0, 1.1],
}

EGO_SMALL_GRID = {
    **SMALL_GRAPH_GRID,
    "graphEmDim": [56, 64, 72, 80, 88, 96],
    "lr": [2e-4, 3e-4, 3.5e-4, 4e-4, 4.5e-4],
    "batchSize": [16, 20, 24],
    "dropout": [0.0, 0.025, 0.05, 0.1],
    "lr_diff": [1e-5, 1.5e-5, 2e-5, 3e-5],
    "flow_steps": [80, 100, 120, 140, 180],
    "dit_hidden_dim": [224, 256, 288, 320],
}




@dataclass(frozen=True)
class SearchSpec:
    dataset: str
    grid: Mapping[str, List]
    guidance_scale: float
    search_method: str = "random"

    @property
    def profile(self) -> str:
        return evaluation_profile(self.dataset)


SEARCH_SPECS = {
    "ego_small": SearchSpec(
        "ego_small", EGO_SMALL_GRID, 1.05, search_method="random"
    ),
    "community_small": SearchSpec(
        "community_small", COMMUNITY_SMALL_GRID, 1.05, search_method="random"
    )
}


def search_spec(dataset: str) -> SearchSpec:
    return SEARCH_SPECS[normalize_benchmark_name(dataset)]


def generate_random_configs(
    grid: Mapping[str, List], num_configs: int, seed: int = 42
) -> List[Dict]:
    rng = random.Random(seed)
    maximum_combinations = math.prod(len(values) for values in grid.values())
    if num_configs > maximum_combinations:
        raise ValueError(
            f"Requested {num_configs} configs from only {maximum_combinations} combinations"
        )

    configurations = []
    seen = set()
    while len(configurations) < num_configs:
        config = {parameter: rng.choice(values) for parameter, values in grid.items()}
        if config["dit_hidden_dim"] % config["dit_num_heads"] != 0:
            continue
        signature = tuple(sorted(config.items()))
        if signature in seen:
            continue
        seen.add(signature)
        configurations.append(config)
    return configurations


def fixed_hyperparameters(spec: SearchSpec, options) -> Dict:
    return {
        "epoch_number": options.epoch_number,
        "epoch_diff": options.epoch_diff,
        "flow_integrator": "heun",
        "directed": False,
        "bfsOrdering": False,
        "node_feat_mode": "struct",
        "lap_pe_dim": 8,
        "encoder_blocks": 4,
        "cond_dim": 64,
        "flow_cond_dropout": 0.1,
        "use_cond_guidance": True,
        "flow_guidance_scale": spec.guidance_scale,
        "seed": options.training_seed,
    }


def _training_args(spec: SearchSpec, options, config: Dict, experiment_id: int):
    from flow_klein.config.standard import parser as training_parser

    fixed = fixed_hyperparameters(spec, options)
    argument_list = [
        "--taskselect", TASK,
        "--dataset", spec.dataset,
        "--device", options.device,
        "--UseGPU", str(options.device.lower() != "cpu"),
    ]
    for parameter, value in {**fixed, **config}.items():
        argument_list.extend([f"--{parameter}", str(value)])
    save_path = Path(options.log_dir) / f"{spec.dataset}_klein_exp{experiment_id}"
    argument_list.extend(["--graph_save_path", str(save_path) + "/"])
    args = training_parser.parse_args(argument_list)

    # add_flags_from_config uses type=bool, for which bool("False") is True.
    for parameter, value in {**fixed, **config}.items():
        if isinstance(value, bool):
            setattr(args, parameter, value)
    args.UseGPU = options.device.lower() != "cpu"
    return args, fixed


def run_experiment(spec: SearchSpec, options, config: Dict, experiment_id: int) -> Dict:
    from flow_klein.training.standard import klein_graphtask

    args, fixed = _training_args(spec, options, config, experiment_id)
    full_config = {**fixed, **config}
    print("\n" + "=" * 72)
    print(f"Experiment {experiment_id + 1}/{options.num_experiments}: {full_config}")
    print("=" * 72)
    started = time.time()
    try:
        metrics = klein_graphtask(args)
        required = (
            ("avg_mmd", "mmd_degree", "mmd_clustering", "mmd_orbit")
            if spec.profile == "degree_clustering_orbit"
            else ("vun", "average_ratio")
        )
        missing = [key for key in required if key not in metrics]
        non_finite = [
            key for key in required
            if key in metrics and not math.isfinite(float(metrics[key]))
        ]
        if missing or non_finite:
            raise ValueError(
                f"Invalid evaluation result; missing={missing}, non_finite={non_finite}"
            )
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


def result_sort_key(profile: str, result: Dict):
    metrics = result["metrics"]
    if profile == "degree_clustering_orbit":
        average_mmd = float(metrics.get("avg_mmd", float("inf")))
        return (average_mmd if math.isfinite(average_mmd) else float("inf"),)
    vun = float(metrics.get("vun", float("-inf")))
    average_ratio = float(metrics.get("average_ratio", float("inf")))
    if not math.isfinite(vun):
        vun = float("-inf")
    if not math.isfinite(average_ratio) or average_ratio < 0:
        average_ratio = float("inf")
    return (
        -vun,
        average_ratio,
    )


def _metric_columns(profile: str):
    if profile == "degree_clustering_orbit":
        return ("mmd_degree", "mmd_clustering", "mmd_orbit", "avg_mmd")
    return (
        "vun",
        "average_ratio",
        "degree_ratio",
        "clustering_ratio",
        "orbit_ratio",
        "spectre_ratio",
        "wavelet_ratio",
    )


def save_results(spec: SearchSpec, options, results: List[Dict]) -> None:
    successful = [result for result in results if result["status"] == "success"]
    failed = [result for result in results if result["status"] != "success"]
    successful.sort(key=lambda result: result_sort_key(spec.profile, result))
    columns = _metric_columns(spec.profile)
    destination = Path(options.results_file)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "w", encoding="utf-8") as handle:
        handle.write("=" * 120 + "\n")
        handle.write("KLEIN FLOW BENCHMARK HYPERPARAMETER SEARCH\n")
        handle.write("=" * 120 + "\n")
        handle.write(f"Dataset: {spec.dataset}\n")
        handle.write(f"Metric profile: {spec.profile}\n")
        handle.write(f"Device: {options.device}\n")
        handle.write(f"Completed experiments: {len(results)}/{options.num_experiments}\n")
        handle.write(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        handle.write("SEARCH SPACE\n")
        for parameter, values in spec.grid.items():
            handle.write(f"  {parameter}: {values}\n")
        handle.write("\n")

        ranking = (
            "Average MMD ascending"
            if spec.profile == "degree_clustering_orbit"
            else "V.U.N. descending, then Average Ratio ascending"
        )
        handle.write(f"RESULTS ({ranking})\n")
        header = f"{'Rank':<6}{'Exp':<6}"
        for column in columns:
            header += f"{column[:17]:<19}"
        header += f"{'Time(s)':<12}Status\n"
        handle.write(header)
        handle.write("-" * 160 + "\n")
        for rank, result in enumerate(successful, 1):
            line = f"{rank:<6}{result['exp_id']:<6}"
            for column in columns:
                value = result["metrics"].get(column, float("nan"))
                line += f"{value:<19.6f}"
            line += f"{result['elapsed_time']:<12.1f}{result['status']}\n"
            handle.write(line)
        for result in failed:
            handle.write(
                f"{'--':<6}{result['exp_id']:<6}"
                + f"{'NaN':<19}" * len(columns)
                + f"{result['elapsed_time']:<12.1f}{result['status']}\n"
            )

        handle.write("\nDETAILED CONFIGURATIONS\n")
        handle.write("=" * 120 + "\n")
        for rank, result in enumerate(successful, 1):
            handle.write(f"Rank {rank} (Exp {result['exp_id']}):\n")
            handle.write(f"  Metrics: {result['metrics']}\n")
            for parameter, value in result["config"].items():
                handle.write(f"  --{parameter} {value}\n")
            handle.write("\n")

        if successful:
            best = successful[0]
            handle.write("BEST CONFIGURATION COMMAND\n")
            handle.write(
                f"python {shlex.quote(str(ROOT / 'scripts' / 'train.py'))} --taskselect {TASK} --dataset {spec.dataset} "
                f"--device {options.device} "
            )
            for parameter, value in best["config"].items():
                handle.write(f"--{parameter} {value} ")
            handle.write("\n")


def build_parser(spec: SearchSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Klein Flow hyperparameter search for {spec.dataset}"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-experiments", type=int, default=60)
    parser.add_argument("--epoch-number", type=int, default=2000)
    parser.add_argument("--epoch-diff", type=int, default=2000)
    parser.add_argument(
        "--results-file", default=None
    )
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=1432)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Download and validate the dataset, then exit without training",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sampled configurations without training dependencies or output writes",
    )
    return parser


def run_search(dataset: str, argv=None) -> None:
    spec = search_spec(dataset)
    options = build_parser(spec).parse_args(argv)
    if options.num_experiments <= 0:
        raise ValueError("--num-experiments must be positive")

    configurations = generate_random_configs(
        spec.grid, options.num_experiments, options.search_seed
    )
    if options.dry_run:
        print(json.dumps(dict(dataset=spec.dataset, pipeline="standard", device=options.device,
                              configurations=[{**fixed_hyperparameters(spec, options), **c}
                                              for c in configurations]), indent=2))
        return

    import torch
    from flow_klein.data.benchmarks_standard import load_benchmark_splits, split_sizes
    from flow_klein.evaluation.standard import preflight_metric_dependencies
    splits = load_benchmark_splits(spec.dataset)
    print(f"Prepared {spec.dataset}: {split_sizes(splits)}")
    if options.prepare_only:
        return
    preflight_metric_dependencies(spec.dataset)
    options.log_dir = str(Path(options.log_dir).resolve() if options.log_dir else search_directory(spec.dataset))
    Path(options.log_dir).mkdir(parents=True, exist_ok=True)
    options.results_file = str(Path(options.results_file).resolve() if options.results_file else
                               Path(options.log_dir) / f"{spec.dataset}_{spec.search_method}.txt")
    Path(options.results_file).parent.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()
    for experiment_id, config in enumerate(configurations):
        result = run_experiment(spec, options, config, experiment_id)
        results.append(result)
        save_results(spec, options, results)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.time() - started
        average = elapsed / len(results)
        remaining = options.num_experiments - len(results)
        print(
            f"Progress {len(results)}/{options.num_experiments}; "
            f"ETA {average * remaining / 60.0:.1f} min"
        )

    save_results(spec, options, results)
    successful = [result for result in results if result["status"] == "success"]
    if successful:
        best = min(successful, key=lambda result: result_sort_key(spec.profile, result))
        print(f"Best experiment: {best['exp_id']}; metrics={best['metrics']}")
