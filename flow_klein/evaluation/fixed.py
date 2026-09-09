"""Dataset-aware metrics ported from GGBall and SimGFM."""

from __future__ import annotations

import os
from flow_klein.paths import ORCA_ROOT
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from flow_klein.data.benchmarks_fixed import normalize_benchmark_name
from .dist_helper import compute_mmd, gaussian_tv
from .spectre import clustering_stats, degree_stats


METRIC_CACHE_VERSION = 1
SMALL_GRAPH_DATASETS = {"ego_small", "community_small"}


def evaluation_profile(dataset: str) -> str:
    """Return the metric profile used by a dataset."""
    try:
        canonical = normalize_benchmark_name(dataset)
    except ValueError:
        return "degree_clustering_spectral"
    if canonical in SMALL_GRAPH_DATASETS:
        return "degree_clustering_orbit"
    return "degree_clustering_spectral"


def normalize_graph(graph: nx.Graph) -> nx.Graph:
    """Create a simple undirected graph without discarding isolated nodes."""
    normalized = nx.Graph(graph)
    normalized.remove_edges_from(nx.selfloop_edges(normalized))
    return nx.convert_node_labels_to_integers(normalized)


def _orca_paths() -> Tuple[Path, Path]:
    orca_dir = ORCA_ROOT
    return orca_dir, orca_dir / "orca"


def validate_orca() -> Path:
    orca_dir, executable = _orca_paths()
    if not executable.is_file():
        source = orca_dir / "orca.cpp"
        raise RuntimeError(
            "ORCA executable is missing. On Linux run: "
            f"g++ -O2 -std=c++11 -o {executable} {source}"
        )
    if platform.system() != "Windows" and not os.access(executable, os.X_OK):
        raise RuntimeError(f"ORCA is not executable: chmod +x {executable}")
    return executable


def preflight_metric_dependencies(dataset: str) -> None:
    """Fail before training when a selected benchmark metric cannot run."""
    profile = evaluation_profile(dataset)
    if profile == "degree_clustering_orbit":
        validate_orca()
        try:
            counts = orca(nx.path_graph(2))
            if counts.shape != (2, 15):
                raise RuntimeError(f"unexpected ORCA output shape {counts.shape}")
        except Exception as exc:
            raise RuntimeError(
                "ORCA preflight failed. Recompile it on the target Linux server with: "
                "g++ -O2 -std=c++11 -o third_party/orca/orca "
                "third_party/orca/orca.cpp"
            ) from exc


def _edge_list_reindexed(graph: nx.Graph) -> List[Tuple[int, int]]:
    node_to_index = {node: index for index, node in enumerate(graph.nodes())}
    return [(node_to_index[u], node_to_index[v]) for u, v in graph.edges()]


def orca(graph: nx.Graph) -> np.ndarray:
    """Return per-node counts for all graphlet orbits on at most four nodes."""
    executable = validate_orca()
    orca_dir = executable.parent
    graph = normalize_graph(graph)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="flow_klein_orca_", dir=orca_dir,
            delete=False, encoding="utf-8"
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
            for u, v in _edge_list_reindexed(graph):
                handle.write(f"{u} {v}\n")

        output = subprocess.check_output(
            [str(executable), "node", "4", str(temporary_path), "std"],
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
        marker = "orbit counts:"
        marker_index = output.find(marker)
        if marker_index < 0:
            raise RuntimeError(f"Unexpected ORCA output: {output[:200]}")
        counts_text = output[marker_index + len(marker) :].strip()
        if not counts_text:
            return np.zeros((graph.number_of_nodes(), 15), dtype=np.int64)
        return np.asarray(
            [[int(value) for value in row.split()] for row in counts_text.splitlines()],
            dtype=np.int64,
        )
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _orbit_features(graphs: Iterable[nx.Graph]) -> np.ndarray:
    features = []
    for graph in graphs:
        if graph.number_of_nodes() == 0:
            continue
        counts = orca(graph)
        denominator = max(counts.shape[0], 1)
        features.append(np.sum(counts, axis=0) / denominator)
    if not features:
        raise ValueError("Orbit metric received no non-empty graphs")
    return np.asarray(features)


def orbit_stats_all(
    graph_ref_list: Iterable[nx.Graph], graph_pred_list: Iterable[nx.Graph]
) -> float:
    ref_features = _orbit_features(graph_ref_list)
    pred_features = _orbit_features(graph_pred_list)
    return float(
        compute_mmd(
            ref_features,
            pred_features,
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )
    )


























def evaluate_external_benchmark(
    dataset: str,
    generated_graphs: Iterable[nx.Graph],
    train_graphs: Iterable[nx.Graph],
    test_graphs: Iterable[nx.Graph],
    cache_dir: Path | str | None = None,
) -> Dict[str, float]:
    canonical = normalize_benchmark_name(dataset)
    generated = [normalize_graph(graph) for graph in generated_graphs]
    train = [normalize_graph(graph) for graph in train_graphs]
    test = [normalize_graph(graph) for graph in test_graphs]
    if not generated or not test:
        raise ValueError(
            f"Invalid evaluation inputs: generated={len(generated)}, test={len(test)}"
        )

    if canonical in SMALL_GRAPH_DATASETS:
        degree = float(degree_stats(test, generated, compute_emd=False))
        clustering = float(clustering_stats(test, generated, compute_emd=False))
        orbit = orbit_stats_all(test, generated)
        return {
            "mmd_degree": degree,
            "mmd_clustering": clustering,
            "mmd_orbit": orbit,
            "avg_mmd": (degree + clustering + orbit) / 3.0,
        }

    raise ValueError("Unsupported fixed benchmark: " + str(dataset))
