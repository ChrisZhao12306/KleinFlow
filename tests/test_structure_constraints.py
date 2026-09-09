import pytest
pytest.importorskip("torch")
pytest.importorskip("dgl")

from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from flow_klein.data.profile import DatasetProfile
from flow_klein.training.structural import (
    _decode_one_topE,
    blend_degree_targets,
    calibrate_edge_probabilities_to_budget,
    checkpoint_selection_key,
    decode_planar_constrained,
    decode_samples_to_graphs,
    decode_tree_constrained,
    project_degree_budget,
    rerank_candidates_by_profile,
)


def _random_symmetric_scores(seed, node_count=64):
    rng = np.random.RandomState(seed)
    scores = rng.normal(size=(node_count, node_count))
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, -10.0)
    return scores


def _uniform_histogram(bin_count=32):
    return np.ones(bin_count, dtype=np.float64) / bin_count


def test_tree_constraint_is_always_a_64_node_63_edge_tree():
    for seed in range(5):
        raw_degrees = np.random.RandomState(seed + 100).uniform(0, 10, size=64)
        target = blend_degree_targets(
            raw_degrees,
            _uniform_histogram(),
            blend=0.75,
            total_degree=126,
            min_degree=1,
        )
        graph = decode_tree_constrained(
            _random_symmetric_scores(seed), target, noise_scale=0.05
        )
        assert graph.number_of_nodes() == 64
        assert graph.number_of_edges() == 63
        assert nx.is_tree(graph)
        assert [graph.degree(node) for node in range(64)] == target.tolist()


def test_planar_constraint_hits_edge_budget_and_rejects_impossible_budget():
    raw_degrees = np.random.RandomState(7).uniform(1, 12, size=64)
    for target_edges in (173, 177, 181):
        target = blend_degree_targets(
            raw_degrees,
            _uniform_histogram(),
            blend=0.5,
            total_degree=2 * target_edges,
            min_degree=1,
        )
        graph = decode_planar_constrained(
            _random_symmetric_scores(target_edges),
            target,
            target_edges=target_edges,
            noise_scale=0.0,
        )
        assert graph.number_of_nodes() == 64
        assert graph.number_of_edges() == target_edges
        assert nx.is_connected(graph)
        assert nx.check_planarity(graph)[0]

    impossible_edges = 3 * 64 - 5
    impossible_target = project_degree_budget(
        np.ones(64), 2 * impossible_edges, min_degree=1, max_degree=63
    )
    try:
        decode_planar_constrained(
            _random_symmetric_scores(9),
            impossible_target,
            target_edges=impossible_edges,
        )
    except ValueError as exc:
        assert "3n-6" in str(exc)
    else:
        raise AssertionError("An E > 3n-6 planar request must fail")


def test_degree_budget_projection_and_profile_blend_are_exact():
    predicted = np.linspace(-2.0, 12.0, 64)
    projected = project_degree_budget(predicted, 354, min_degree=1, max_degree=63)
    assert projected.dtype == np.int64
    assert int(projected.sum()) == 354
    assert projected.min() >= 1
    assert projected.max() <= 63

    model_only = blend_degree_targets(
        predicted, _uniform_histogram(), 0.0, 126, min_degree=1
    )
    profile_only = blend_degree_targets(
        predicted, _uniform_histogram(), 1.0, 126, min_degree=1
    )
    assert int(model_only.sum()) == 126
    assert int(profile_only.sum()) == 126
    assert not np.array_equal(model_only, profile_only)


def test_soft_edge_calibration_matches_each_target_edge_budget():
    logits = torch.randn(2, 6, 6, requires_grad=True)
    pair_mask = torch.ones_like(logits)
    diagonal = torch.arange(6)
    pair_mask[:, diagonal, diagonal] = 0.0
    target_edges = torch.tensor([5.0, 12.0])
    calibrated = calibrate_edge_probabilities_to_budget(
        logits, pair_mask, target_edges, temperature=0.5
    )
    realized = torch.triu(calibrated, diagonal=1).sum(dim=(1, 2))
    assert torch.allclose(realized, target_edges, atol=1e-4, rtol=0.0)
    calibrated.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_condition_profile_bank_follows_current_graph_order():
    graphs = [
        nx.path_graph(8),
        nx.cycle_graph(8),
        nx.star_graph(7),
    ]
    adjs = [sp.csr_matrix(nx.to_numpy_array(graph)) for graph in graphs]
    profile = DatasetProfile().fit(adjs)
    reordered = [adjs[2], adjs[0], adjs[1]]
    scalar_bank, histogram_bank = profile.aligned_bank(reordered)
    raw = profile.destandardize_scalar(scalar_bank)

    assert raw[:, 1].round().to(torch.int64).tolist() == [7, 7, 8]
    assert histogram_bank.shape == (3, profile.n_bins)
    assert torch.allclose(histogram_bank.sum(dim=1), torch.ones(3))


def test_constraint_noise_is_reproducible_with_fixed_seed():
    scores = _random_symmetric_scores(11)
    target = blend_degree_targets(
        np.linspace(1, 8, 64), _uniform_histogram(), 0.75, 126, min_degree=1
    )
    np.random.seed(1432)
    first = decode_tree_constrained(scores, target, noise_scale=0.1)
    np.random.seed(1432)
    second = decode_tree_constrained(scores, target, noise_scale=0.1)
    assert set(first.edges()) == set(second.edges())


def test_checkpoint_selection_is_vun_first_then_proxy_ratio():
    higher_vun = {"vun": 0.9, "proxy_ratio": 100.0}
    lower_vun = {"vun": 0.8, "proxy_ratio": 0.1}
    same_vun_better_ratio = {"vun": 0.9, "proxy_ratio": 2.0}
    assert checkpoint_selection_key(higher_vun, "planar") < checkpoint_selection_key(
        lower_vun, "planar"
    )
    assert checkpoint_selection_key(
        same_vun_better_ratio, "tree"
    ) < checkpoint_selection_key(higher_vun, "tree")


def test_unique_reranking_does_not_consult_training_graphs():
    candidates = [nx.path_graph(8), nx.path_graph(8), nx.cycle_graph(8)]
    profile = DatasetProfile().fit(
        [sp.csr_matrix(nx.to_numpy_array(graph)) for graph in candidates]
    )
    selected = rerank_candidates_by_profile(
        candidates, profile, target_n=2, prefer_unique=True
    )
    assert len(selected) == 2
    assert not nx.is_isomorphic(selected[0], selected[1])


def test_grid_top_e_decoding_is_bit_exact_when_constraint_is_none():
    probabilities = torch.tensor(
        [
            [0.0, 0.90, 0.80, 0.10],
            [0.90, 0.0, 0.70, 0.20],
            [0.80, 0.70, 0.0, 0.30],
            [0.10, 0.20, 0.30, 0.0],
        ],
        dtype=torch.float32,
    )
    node_probabilities = torch.tensor([0.9, 0.8, 0.7, 0.6])
    stats = torch.tensor([4.0, 2.0])
    expected, _ = _decode_one_topE(
        probabilities,
        node_probabilities,
        stats,
        None,
        False,
        4,
        1,
        4,
        0.10,
        2,
    )
    assert np.array_equal(
        expected,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )

    args = SimpleNamespace(
        decode_mode="topE",
        structure_constraint="none",
        connectivity_repair=False,
        topE_tolerance=0.10,
        topE_abs_tol=2,
        edge_budget_rel_tol=0.05,
        edge_budget_abs_tol=2,
        directed=False,
    )
    decoded = decode_samples_to_graphs(
        args,
        probabilities.unsqueeze(0),
        torch.logit(node_probabilities).unsqueeze(0),
        stats.unsqueeze(0),
    )
    assert len(decoded) == 1
    # Cleanup drops the isolated slot and keeps the remaining Top-E edges.
    assert np.array_equal(nx.to_numpy_array(decoded[0]), expected[:3, :3])
