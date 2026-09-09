"""SimGFM V.U.N. and Ratio metrics for external graph benchmarks."""

from __future__ import annotations

import os
from flow_klein.paths import ORCA_ROOT, DATA_ROOT
import pickle
import platform
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import networkx as nx
import numpy as np

from flow_klein.data.benchmarks_v0901 import normalize_benchmark_name
from .dist_helper import compute_mmd, gaussian_tv
from .spectre import clustering_stats, degree_stats, spectral_stats


METRIC_CACHE_VERSION = 1
VUN_RATIO_DATASETS = {"planar", "tree"}
RATIO_METRICS = ("degree", "clustering", "orbit", "spectre", "wavelet")


def evaluation_profile(dataset: str) -> str:
    """Return the final-test metric profile used by a dataset."""
    try:
        canonical = normalize_benchmark_name(dataset)
    except ValueError:
        return "degree_clustering_spectral"
    if canonical in VUN_RATIO_DATASETS:
        return "vun_ratio"
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
        # Executable bits are commonly lost when the repository is copied from
        # Windows to a Linux server. Restore them during preflight so every
        # launcher and direct Python invocation behaves consistently.
        try:
            executable.chmod(
                executable.stat().st_mode
                | stat.S_IXUSR
                | stat.S_IXGRP
                | stat.S_IXOTH
            )
        except OSError as exc:
            raise RuntimeError(
                f"ORCA is not executable and chmod failed for {executable}: {exc}"
            ) from exc
        if not os.access(executable, os.X_OK):
            raise RuntimeError(f"ORCA is not executable: chmod +x {executable}")
    return executable


def preflight_metric_dependencies(dataset: str) -> None:
    """Fail before training when a benchmark metric cannot run."""
    if evaluation_profile(dataset) != "vun_ratio":
        return

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

    try:
        import pygsp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "SimGFM Ratio requires PyGSP. Install PyGSP==0.5.1."
        ) from exc



def _edge_list_reindexed(graph: nx.Graph) -> List[Tuple[int, int]]:
    node_to_index = {node: index for index, node in enumerate(graph.nodes())}
    return [(node_to_index[u], node_to_index[v]) for u, v in graph.edges()]


def orca(graph: nx.Graph) -> np.ndarray:
    """Return per-node counts for all graphlet orbits on at most four nodes."""
    executable = validate_orca()
    graph = normalize_graph(graph)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="flow_klein_orca_",
            dir=executable.parent,
            delete=False,
            encoding="utf-8",
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
        features.append(np.sum(counts, axis=0) / max(counts.shape[0], 1))
    if not features:
        raise ValueError("Orbit metric received no non-empty graphs")
    return np.asarray(features)


def orbit_stats_all(
    graph_ref_list: Iterable[nx.Graph], graph_pred_list: Iterable[nx.Graph]
) -> float:
    return float(
        compute_mmd(
            _orbit_features(graph_ref_list),
            _orbit_features(graph_pred_list),
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )
    )


def _compute_eigendecompositions(graphs: Iterable[nx.Graph]):
    eigenvalues = []
    eigenvectors = []
    for graph in graphs:
        laplacian = nx.normalized_laplacian_matrix(graph).todense()
        try:
            values, vectors = np.linalg.eigh(laplacian)
        except Exception:
            values = np.zeros(laplacian.shape[0])
            vectors = np.zeros(laplacian.shape)
        eigenvalues.append(values)
        eigenvectors.append(vectors)
    return eigenvalues, eigenvectors


def _wavelet_histogram(eigenvectors, eigenvalues, filters, bound: float) -> np.ndarray:
    evaluated = filters.evaluate(eigenvalues)
    operators = np.asarray(
        [eigenvectors @ np.diag(response) @ eigenvectors.T for response in evaluated]
    )
    norms = np.sum(operators ** 2, axis=2)
    return np.asarray(
        [np.histogram(values, range=(0, bound), bins=100)[0] for values in norms]
    ).flatten()


def wavelet_stats(
    graph_ref_list: Iterable[nx.Graph], graph_pred_list: Iterable[nx.Graph]
) -> float:
    import pygsp as pg

    class DummyNormalizedGraph:
        lmax = 2

    filters = pg.filters.Abspline(DummyNormalizedGraph, 12)
    bound = float(np.max(filters.evaluate(np.arange(0, 2, 0.01))))
    ref_values, ref_vectors = _compute_eigendecompositions(graph_ref_list)
    pred_values, pred_vectors = _compute_eigendecompositions(graph_pred_list)
    ref_samples = [
        _wavelet_histogram(vectors, values, filters, bound)
        for values, vectors in zip(ref_values, ref_vectors)
    ]
    pred_samples = [
        _wavelet_histogram(vectors, values, filters, bound)
        for values, vectors in zip(pred_values, pred_vectors)
    ]
    return float(compute_mmd(ref_samples, pred_samples, kernel=gaussian_tv))


def structural_metrics(
    reference_graphs: List[nx.Graph], generated_graphs: List[nx.Graph]
) -> Dict[str, float]:
    """Compute SimGFM's five base distances with compute_emd=False."""
    return {
        "degree": float(
            degree_stats(reference_graphs, generated_graphs, compute_emd=False)
        ),
        "clustering": float(
            clustering_stats(reference_graphs, generated_graphs, compute_emd=False)
        ),
        "orbit": orbit_stats_all(reference_graphs, generated_graphs),
        "spectre": float(
            spectral_stats(reference_graphs, generated_graphs, compute_emd=False)
        ),
        "wavelet": wavelet_stats(reference_graphs, generated_graphs),
    }


def compute_ratios(
    generated_metrics: Mapping[str, float], reference_metrics: Mapping[str, float]
) -> Dict[str, float]:
    ratios = {}
    for key in RATIO_METRICS:
        reference = round(float(reference_metrics[key]), 4)
        if reference == 0.0:
            continue
        ratios[f"{key}_ratio"] = float(generated_metrics[key]) / reference
    ratios["average_ratio"] = (
        float(np.mean(list(ratios.values()))) if ratios else -1.0
    )
    return ratios


def is_planar_graph(graph: nx.Graph) -> bool:
    return (
        graph.number_of_nodes() > 0
        and nx.is_connected(graph)
        and nx.check_planarity(graph)[0]
    )


def is_tree_graph(graph: nx.Graph) -> bool:
    return graph.number_of_nodes() > 0 and nx.is_tree(graph)




def _validity_function(dataset: str) -> Callable[[nx.Graph], bool]:
    return {
        "planar": is_planar_graph,
        "tree": is_tree_graph,
    }[normalize_benchmark_name(dataset)]


def compute_vun(
    generated_graphs: List[nx.Graph],
    train_graphs: List[nx.Graph],
    validity_function: Callable[[nx.Graph], bool],
) -> Dict[str, float]:
    """Compute SimGFM validity, uniqueness, novelty, and V.U.N."""
    total = len(generated_graphs)
    if total == 0:
        return {
            "frac_valid": 0.0,
            "frac_unique": 0.0,
            "frac_non_iso": 0.0,
            "frac_unique_non_iso": 0.0,
            "vun": 0.0,
        }

    unique_graphs = []
    duplicate_count = 0
    train_isomorphic_count = 0
    unique_novel_valid_count = 0
    valid_count = 0
    non_isomorphic_count = 0

    for generated in generated_graphs:
        is_valid = validity_function(generated)
        if is_valid:
            valid_count += 1
        is_train_graph = any(
            nx.faster_could_be_isomorphic(generated, train)
            and nx.is_isomorphic(generated, train)
            for train in train_graphs
        )
        if not is_train_graph:
            non_isomorphic_count += 1

        is_duplicate = any(nx.is_isomorphic(generated, old) for old in unique_graphs)
        if is_duplicate:
            duplicate_count += 1
            continue
        unique_graphs.append(generated)

        if is_train_graph:
            train_isomorphic_count += 1
        elif is_valid:
            unique_novel_valid_count += 1

    return {
        "frac_valid": valid_count / float(total),
        "frac_unique": (total - duplicate_count) / float(total),
        "frac_non_iso": non_isomorphic_count / float(total),
        "frac_unique_non_iso": (
            total - duplicate_count - train_isomorphic_count
        ) / float(total),
        "vun": unique_novel_valid_count / float(total),
    }


def _reference_cache_path(dataset: str, cache_dir: Path | str | None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else (
        DATA_ROOT / "benchmarks"
    )
    return root / normalize_benchmark_name(dataset) / "reference_metrics_v1.pkl"


def reference_metrics(
    dataset: str,
    train_graphs: List[nx.Graph],
    test_graphs: List[nx.Graph],
    cache_dir: Path | str | None = None,
) -> Dict[str, float]:
    path = _reference_cache_path(dataset, cache_dir)
    canonical = normalize_benchmark_name(dataset)
    if path.is_file():
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if (
            payload.get("version") == METRIC_CACHE_VERSION
            and payload.get("dataset") == canonical
        ):
            return payload["metrics"]

    metrics = structural_metrics(train_graphs, test_graphs)
    payload = {
        "version": METRIC_CACHE_VERSION,
        "dataset": canonical,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temporary, path)
    return metrics


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

    generated_metrics = structural_metrics(test, generated)
    baseline_metrics = reference_metrics(canonical, train, test, cache_dir)
    result = dict(generated_metrics)
    result.update(compute_vun(generated, train, _validity_function(canonical)))
    result.update(compute_ratios(generated_metrics, baseline_metrics))
    return result
