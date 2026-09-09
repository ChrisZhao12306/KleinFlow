"""Dataset registry for the external graph-generation benchmarks.

The loaders in this module intentionally keep the train/validation/test splits
used by GGBall and SimGFM.  Raw source files are downloaded once and then read
from ``data/benchmarks`` so a prepared checkout can be used offline.
"""

from __future__ import annotations

import os
from flow_klein.paths import DATA_ROOT
import pickle
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


BENCHMARK_ALIASES = {
    "ego_small": "ego_small",
    "ego-small": "ego_small",
    "community_small": "community_small",
    "community-small": "community_small",
    "comm20": "community_small"
}

BENCHMARK_URLS = {
    "ego_small": (
        "ego_small.pkl",
        "https://raw.githubusercontent.com/harryjo97/GDSS/master/data/ego_small.pkl",
    ),
    "community_small": (
        "community_12_21_100.pt",
        "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt",
    )
}

EXPECTED_SPLIT_SIZES = {
    "ego_small": {"train": 128, "val": 32, "test": 40}
}


def normalize_benchmark_name(name: str) -> str:
    """Return the canonical benchmark name or raise a helpful error."""
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
        return destination

    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Flow-Klein benchmark loader"})
        with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty dataset file from {url}")
        os.replace(temporary, destination)
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


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    elif torch.is_tensor(graph):
        adjacency = sp.csr_matrix(graph.detach().cpu().numpy(), dtype=np.float32)
    elif sp.issparse(graph):
        adjacency = sp.csr_matrix(graph, dtype=np.float32)
    else:
        adjacency = sp.csr_matrix(np.asarray(graph, dtype=np.float32))

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected a square adjacency matrix, got {adjacency.shape}")
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    if adjacency.nnz:
        adjacency.data[:] = 1.0
    return adjacency


def _validate_split_mapping(
    name: str, raw_splits: Mapping[str, Iterable], expected: Mapping[str, int] | None
) -> Dict[str, List[sp.csr_matrix]]:
    missing = {"train", "val", "test"}.difference(raw_splits)
    if missing:
        raise ValueError(f"Dataset {name} is missing splits: {sorted(missing)}")

    splits = {
        split: [_to_csr_adjacency(graph) for graph in raw_splits[split]]
        for split in ("train", "val", "test")
    }
    if expected is not None:
        actual = {split: len(graphs) for split, graphs in splits.items()}
        if actual != dict(expected):
            raise ValueError(
                f"Unexpected {name} split sizes: {actual}; expected {dict(expected)}"
            )
    return splits


def _ggball_split(graphs: List, index_space_size: int = 200) -> Dict[str, List]:
    """Reproduce GGBall's seed-0 split, including comm20's 200-index quirk."""
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(index_space_size, generator=generator)
    test_len = int(round(index_space_size * 0.2))
    train_len = int(round((index_space_size - test_len) * 0.8))
    train_indices = set(indices[:train_len].tolist())
    val_indices = set(indices[train_len : index_space_size - test_len].tolist())
    test_indices = set(indices[index_space_size - test_len :].tolist())

    split = {"train": [], "val": [], "test": []}
    for index, graph in enumerate(graphs):
        if index in train_indices:
            split["train"].append(graph)
        elif index in val_indices:
            split["val"].append(graph)
        elif index in test_indices:
            split["test"].append(graph)
        else:
            raise ValueError(f"GGBall split did not assign graph index {index}")
    return split


def load_benchmark_splits(
    name: str, cache_dir: Path | str | None = None
) -> Dict[str, List[sp.csr_matrix]]:
    """Load a benchmark as fixed train/val/test SciPy CSR adjacency lists."""
    canonical = normalize_benchmark_name(name)
    raw_path = benchmark_raw_path(canonical, cache_dir=cache_dir)

    if canonical == "ego_small":
        with open(raw_path, "rb") as handle:
            graphs = pickle.load(handle)
        if len(graphs) != 200:
            raise ValueError(f"Expected 200 ego_small graphs, found {len(graphs)}")
        raw_splits = _ggball_split(list(graphs), index_space_size=200)
    elif canonical == "community_small":
        payload = _torch_load(raw_path)
        if not isinstance(payload, (tuple, list)) or len(payload) < 1:
            raise ValueError("Unexpected GGBall comm20 payload")
        graphs = list(payload[0])
        if len(graphs) != 100:
            raise ValueError(f"Expected 100 comm20 graphs, found {len(graphs)}")
        raw_splits = _ggball_split(graphs, index_space_size=200)
    else:
        with open(raw_path, "rb") as handle:
            raw_splits = pickle.load(handle)

    return _validate_split_mapping(
        canonical, raw_splits, EXPECTED_SPLIT_SIZES.get(canonical)
    )


def split_sizes(splits: Mapping[str, Iterable]) -> Dict[str, int]:
    return {split: len(splits[split]) for split in ("train", "val", "test")}


def prepare_benchmarks(
    names: Iterable[str] | None = None, cache_dir: Path | str | None = None
) -> Dict[str, Dict[str, int]]:
    selected = list(names) if names is not None else list(BENCHMARK_URLS)
    summary = {}
    for name in selected:
        canonical = normalize_benchmark_name(name)
        summary[canonical] = split_sizes(load_benchmark_splits(canonical, cache_dir))
    return summary
