import pytest
pytest.importorskip("scipy")
pytest.importorskip("pyemd")

import pickle
import platform
from unittest import mock

import networkx as nx
import numpy as np
import pytest

from flow_klein.evaluation import v0901 as metrics


def test_profile_routing_keeps_old_datasets_unchanged():
    assert metrics.evaluation_profile("planar") == "vun_ratio"
    assert metrics.evaluation_profile("tree") == "vun_ratio"
    assert metrics.evaluation_profile("MUTAG") == "degree_clustering_spectral"


def test_planar_and_tree_validity():
    assert metrics.is_planar_graph(nx.cycle_graph(4))
    assert not metrics.is_planar_graph(nx.disjoint_union(nx.path_graph(2), nx.path_graph(2)))
    assert not metrics.is_planar_graph(nx.complete_bipartite_graph(3, 3))
    assert metrics.is_tree_graph(nx.path_graph(5))
    assert not metrics.is_tree_graph(nx.cycle_graph(5))


def test_vun_uses_all_generated_graphs_as_denominator():
    train = [nx.path_graph(3)]
    generated = [nx.path_graph(4), nx.path_graph(4), nx.cycle_graph(3)]
    result = metrics.compute_vun(generated, train, nx.is_tree)
    assert result["frac_valid"] == pytest.approx(2 / 3)
    assert result["frac_unique"] == pytest.approx(2 / 3)
    assert result["frac_non_iso"] == pytest.approx(1.0)
    assert result["frac_unique_non_iso"] == pytest.approx(2 / 3)
    assert result["vun"] == pytest.approx(1 / 3)


def test_ratio_rounds_reference_to_four_decimals():
    generated = {key: 1.0 for key in metrics.RATIO_METRICS}
    reference = {key: 0.33334 for key in metrics.RATIO_METRICS}
    result = metrics.compute_ratios(generated, reference)
    expected = 1.0 / 0.3333
    assert result["degree_ratio"] == pytest.approx(expected)
    assert result["average_ratio"] == pytest.approx(expected)


def test_zero_rounded_reference_ratios_are_undefined_and_excluded():
    generated = {key: 1.0 for key in metrics.RATIO_METRICS}
    reference = {
        "degree": 0.5,
        "clustering": 0.0,
        "orbit": 5.878e-7,
        "spectre": 0.25,
        "wavelet": 0.2,
    }
    result = metrics.compute_ratios(generated, reference)
    assert "clustering_ratio" not in result
    assert "orbit_ratio" not in result
    assert result["average_ratio"] == pytest.approx((2.0 + 4.0 + 5.0) / 3.0)




def test_reference_metric_cache_is_reused(tmp_path):
    expected = {key: float(index + 1) for index, key in enumerate(metrics.RATIO_METRICS)}
    with mock.patch.object(metrics, "structural_metrics", return_value=expected) as compute:
        first = metrics.reference_metrics(
            "tree", [nx.path_graph(3)], [nx.path_graph(4)], cache_dir=tmp_path
        )
        second = metrics.reference_metrics(
            "tree", [nx.path_graph(3)], [nx.path_graph(4)], cache_dir=tmp_path
        )
    assert first == second == expected
    assert compute.call_count == 1
    cache = tmp_path / "tree" / "reference_metrics_v1.pkl"
    with open(cache, "rb") as handle:
        assert pickle.load(handle)["version"] == metrics.METRIC_CACHE_VERSION


def test_orca_smoke_and_temporary_cleanup():
    if platform.system() == "Windows":
        pytest.skip("Bundled ORCA executable is compiled on the Linux server")
    try:
        executable = metrics.validate_orca()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    before = set(executable.parent.glob("flow_klein_orca_*.txt"))
    counts = metrics.orca(nx.path_graph(4))
    after = set(executable.parent.glob("flow_klein_orca_*.txt"))
    assert counts.shape == (4, 15)
    assert np.isfinite(counts).all()
    assert after == before


def test_wavelet_smoke_when_pygsp_is_installed():
    pytest.importorskip("pygsp")
    value = metrics.wavelet_stats([nx.path_graph(4)], [nx.cycle_graph(4)])
    assert np.isfinite(value)
