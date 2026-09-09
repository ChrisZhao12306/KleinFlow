import pytest
pytest.importorskip("scipy")

import pickle
from pathlib import Path
from unittest import mock

import networkx as nx

from flow_klein.data.benchmarks_v0901 import (
    _download_once,
    load_benchmark_splits,
    normalize_benchmark_name,
    split_sizes,
)


def test_downloader_uses_curl_and_reuses_cache(tmp_path: Path):
    destination = tmp_path / "planar.pkl"

    def fake_run(command, check):
        assert command[1:4] == ["-fL", "--retry", "5"]
        assert check is True
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"pickle-data")

    with mock.patch("flow_klein.data.benchmarks_v0901.shutil.which", return_value="/usr/bin/curl"), \
            mock.patch("flow_klein.data.benchmarks_v0901.subprocess.run", side_effect=fake_run) as run:
        assert _download_once("https://example.test/planar.pkl", destination) == destination
        assert _download_once("https://example.test/planar.pkl", destination) == destination

    assert destination.read_bytes() == b"pickle-data"
    assert run.call_count == 1


def test_aliases_are_case_insensitive():
    assert normalize_benchmark_name("PLANAR") == "planar"
    assert normalize_benchmark_name("Tree") == "tree"


def test_fixed_pickle_split_and_csr_normalization(tmp_path: Path):
    raw = {
        "train": [nx.path_graph(size) for size in range(1, 129)],
        "val": [nx.path_graph(size) for size in range(129, 161)],
        "test": [nx.path_graph(size) for size in range(161, 201)],
    }
    destination = tmp_path / "planar" / "planar.pkl"
    destination.parent.mkdir(parents=True)
    with open(destination, "wb") as handle:
        pickle.dump(raw, handle)

    splits = load_benchmark_splits("planar", cache_dir=tmp_path)
    assert split_sizes(splits) == {"train": 128, "val": 32, "test": 40}
    assert {adj.shape[0] for adj in splits["train"]}.isdisjoint(
        {adj.shape[0] for adj in splits["val"]}
    )
    assert {adj.shape[0] for adj in splits["train"]}.isdisjoint(
        {adj.shape[0] for adj in splits["test"]}
    )
    assert all(adj.diagonal().sum() == 0 for part in splits.values() for adj in part)
    assert all((adj != adj.transpose()).nnz == 0 for part in splits.values() for adj in part)
