"""Fixed-split loaders for the SimGFM graph-generation benchmarks."""

from __future__ import annotations

import os
from flow_klein.paths import DATA_ROOT
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import networkx as nx
import numpy as np
import scipy.sparse as sp


BENCHMARK_ALIASES = {
    "planar": "planar",
    "tree": "tree"
}

BENCHMARK_URLS = {
    "planar": (
        "planar.pkl",
        "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/planar.pkl",
    ),
    "tree": (
        "tree.pkl",
        "https://raw.githubusercontent.com/AndreasBergmeister/graph-generation/main/data/tree.pkl",
    )
}

EXPECTED_SPLIT_SIZES = {
    name: {"train": 128, "val": 32, "test": 40}
    for name in BENCHMARK_URLS
}


def normalize_benchmark_name(name: str) -> str:
    """Return the canonical lowercase benchmark name."""
    key = str(name).strip().lower()
    if key not in BENCHMARK_ALIASES:
        supported = ", ".join(sorted(BENCHMARK_URLS))
        raise ValueError(f"Unknown benchmark dataset {name!r}. Supported: {supported}")
    return BENCHMARK_ALIASES[key]


def is_benchmark_dataset(name: str) -> bool:
    return str(name).strip().lower() in BENCHMARK_ALIASES


def default_cache_dir() -> Path:
    return DATA_ROOT / "benchmarks"


def _download_once(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"[benchmark-data] cached: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError(
                "The benchmark downloader requires curl. Install curl or "
                "copy the benchmark pickle files into data/benchmarks manually."
            )
        print(f"[benchmark-data] downloading with curl: {url}", flush=True)
        subprocess.run(
            [
                curl,
                "-fL",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--connect-timeout",
                "10",
                "--max-time",
                "300",
                "-o",
                str(temporary),
                url,
            ],
            check=True,
        )
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty dataset file from {url}")
        os.replace(temporary, destination)
        print(
            f"[benchmark-data] saved {destination} "
            f"({destination.stat().st_size} bytes)",
            flush=True,
        )
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def benchmark_raw_path(name: str, cache_dir: Path | str | None = None) -> Path:
    canonical = normalize_benchmark_name(name)
    filename, url = BENCHMARK_URLS[canonical]
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return _download_once(url, root / canonical / filename)


def _to_csr_adjacency(graph) -> sp.csr_matrix:
    if isinstance(graph, nx.Graph):
        nodes = list(graph.nodes())
        if hasattr(nx, "to_scipy_sparse_array"):
            adjacency = nx.to_scipy_sparse_array(
                graph, nodelist=nodes, dtype=np.float32, format="csr"
            )
        else:
            adjacency = nx.to_scipy_sparse_matrix(
                graph, nodelist=nodes, dtype=np.float32, format="csr"
            )
        adjacency = sp.csr_matrix(adjacency)
    elif sp.issparse(graph):
        adjacency = sp.csr_matrix(graph, dtype=np.float32)
    else:
        adjacency = sp.csr_matrix(np.asarray(graph, dtype=np.float32))

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got {adjacency.shape}")
    adjacency = adjacency.maximum(adjacency.transpose()).tolil()
    adjacency.setdiag(0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    if adjacency.nnz:
        adjacency.data[:] = 1.0
    return adjacency


def _validate_split_mapping(
    name: str, raw_splits: Mapping[str, Iterable]
) -> Dict[str, List[sp.csr_matrix]]:
    missing = {"train", "val", "test"}.difference(raw_splits)
    if missing:
        raise ValueError(f"Dataset {name} is missing splits: {sorted(missing)}")

    splits = {
        split: [_to_csr_adjacency(graph) for graph in raw_splits[split]]
        for split in ("train", "val", "test")
    }
    actual = split_sizes(splits)
    expected = EXPECTED_SPLIT_SIZES[name]
    if actual != expected:
        raise ValueError(
            f"Unexpected {name} split sizes: {actual}; expected {expected}"
        )
    return splits


def load_benchmark_splits(
    name: str, cache_dir: Path | str | None = None
) -> Dict[str, List[sp.csr_matrix]]:
    """Load the upstream 128/32/40 train/validation/test split."""
    canonical = normalize_benchmark_name(name)
    raw_path = benchmark_raw_path(canonical, cache_dir=cache_dir)
    with open(raw_path, "rb") as handle:
        raw_splits = pickle.load(handle)
    return _validate_split_mapping(canonical, raw_splits)


def split_sizes(splits: Mapping[str, Iterable]) -> Dict[str, int]:
    return {split: len(splits[split]) for split in ("train", "val", "test")}


def prepare_benchmarks(
    names: Iterable[str] | None = None, cache_dir: Path | str | None = None
) -> Dict[str, Dict[str, int]]:
    selected = list(names) if names is not None else list(BENCHMARK_URLS)
    summary = {}
    for name in selected:
        canonical = normalize_benchmark_name(name)
        print(f"[benchmark-data] preparing {canonical}", flush=True)
        sizes = split_sizes(load_benchmark_splits(canonical, cache_dir))
        summary[canonical] = sizes
        print(
            f"{canonical}: train={sizes['train']} "
            f"val={sizes['val']} test={sizes['test']}",
            flush=True,
        )
    return summary
