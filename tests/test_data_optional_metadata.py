import pytest
pytest.importorskip("torch")
pytest.importorskip("dgl")

import scipy.sparse as sp

from flow_klein.data.structural import Datasets


def test_size_filter_preserves_missing_labels_and_features():
    dataset = object.__new__(Datasets)
    adjacencies = [sp.eye(2, format="csr"), sp.eye(4, format="csr")]

    filtered, labels, features = dataset.remove_largergraphs(
        adjacencies, labels=None, Xs=None, max_size=2
    )

    assert len(filtered) == 1
    assert labels is None
    assert features is None


def test_size_filter_keeps_aligned_optional_metadata():
    dataset = object.__new__(Datasets)
    adjacencies = [sp.eye(2, format="csr"), sp.eye(4, format="csr")]

    filtered, labels, features = dataset.remove_largergraphs(
        adjacencies,
        labels=[10, 20],
        Xs=["small", "large"],
        max_size=2,
    )

    assert len(filtered) == 1
    assert labels == [10]
    assert features == ["small"]
