"""
Klein GraphTask dataset profiles and per-dataset default recipes.

DatasetProfile collects train-core graph statistics for:
  - structural conditioning (encoder side)
  - sampling-time stats bank
  - candidate reranking after generation

DATASET_DEFAULTS provides per-dataset starting points for the new pipeline.
CLI flags always override these defaults.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch


_DEGREE_HIST_BINS = 32


def _to_dense_array(adj) -> np.ndarray:
    """Convert scipy sparse / dense / torch matrix to a square dense numpy array."""
    if isinstance(adj, np.ndarray):
        a = adj
    elif isinstance(adj, torch.Tensor):
        a = adj.detach().cpu().numpy()
    elif sp.issparse(adj):
        a = adj.toarray()
    else:
        a = np.asarray(adj)
    a = (a > 0).astype(np.float32)
    np.fill_diagonal(a, 0.0)
    return a


def _scalar_stats_from_dense(a: np.ndarray) -> np.ndarray:
    """Return (n, m, density, avg_deg, deg_mom2, deg_mom3, deg_mom4, max_deg).
    Length 8."""
    n = int(a.shape[0])
    if n <= 1:
        return np.zeros(8, dtype=np.float32)
    deg = a.sum(axis=-1)
    m = float(deg.sum() / 2.0)
    max_edges = n * (n - 1) / 2.0
    density = float(m / max_edges) if max_edges > 0 else 0.0
    avg_deg = float(deg.mean())
    deg_mom2 = float((deg ** 2).mean())
    deg_mom3 = float((deg ** 3).mean())
    deg_mom4 = float((deg ** 4).mean())
    max_deg = float(deg.max())
    return np.array(
        [n, m, density, avg_deg, deg_mom2, deg_mom3, deg_mom4, max_deg],
        dtype=np.float32,
    )


def _degree_histogram(a: np.ndarray, n_bins: int = _DEGREE_HIST_BINS) -> np.ndarray:
    """Normalized degree histogram on [0, max_possible_degree]."""
    n = int(a.shape[0])
    if n <= 1:
        h = np.zeros(n_bins, dtype=np.float32)
        h[0] = 1.0
        return h
    deg = a.sum(axis=-1).astype(np.float32)
    # bins normalized over [0, n-1], inclusive at right edge.
    edges = np.linspace(0.0, max(float(n - 1), 1.0), n_bins + 1)
    hist, _ = np.histogram(deg, bins=edges)
    total = float(hist.sum())
    if total <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    return (hist / total).astype(np.float32)


def _stats_from_graph(g: nx.Graph, n_bins: int = _DEGREE_HIST_BINS):
    n = g.number_of_nodes()
    if n <= 0:
        return np.zeros(8, dtype=np.float32), np.zeros(n_bins, dtype=np.float32)
    a = nx.to_numpy_array(g, dtype=np.float32)
    np.fill_diagonal(a, 0.0)
    return _scalar_stats_from_dense(a), _degree_histogram(a, n_bins)


class DatasetProfile:
    """Train-core-only statistics container.

    Fields populated by .fit():
      - scalar_mean, scalar_std (length 8 vectors)
      - hist_mean (length n_bins)
      - n_min, n_max, m_min, m_max
      - scalar_bank (N, 8) and hist_bank (N, n_bins): standardized scalars + raw histograms.

    Persistence is via state_dict / load_state_dict (plain tensors).
    """

    SCALAR_KEYS = (
        "node_count",
        "edge_count",
        "density",
        "avg_degree",
        "degree_moment_2",
        "degree_moment_3",
        "degree_moment_4",
        "max_degree",
    )

    def __init__(self, n_bins: int = _DEGREE_HIST_BINS):
        self.n_bins = n_bins
        self.scalar_mean: Optional[torch.Tensor] = None
        self.scalar_std: Optional[torch.Tensor] = None
        self.hist_mean: Optional[torch.Tensor] = None
        self.scalar_bank: Optional[torch.Tensor] = None
        self.hist_bank: Optional[torch.Tensor] = None
        self.n_min: int = 0
        self.n_max: int = 0
        self.m_min: int = 0
        self.m_max: int = 0
        self._fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, train_core_adjs: Iterable) -> "DatasetProfile":
        scalars = []
        hists = []
        for adj in train_core_adjs:
            a = _to_dense_array(adj)
            scalars.append(_scalar_stats_from_dense(a))
            hists.append(_degree_histogram(a, self.n_bins))
        if not scalars:
            raise ValueError("DatasetProfile.fit got an empty train-core list.")

        scalar_arr = np.stack(scalars, axis=0)   # (N, 8)
        hist_arr = np.stack(hists, axis=0)        # (N, n_bins)

        mean = scalar_arr.mean(axis=0)
        std = scalar_arr.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        std_scalar_bank = ((scalar_arr - mean) / std).astype(np.float32)

        self.scalar_mean = torch.from_numpy(mean.astype(np.float32))
        self.scalar_std = torch.from_numpy(std.astype(np.float32))
        self.hist_mean = torch.from_numpy(hist_arr.mean(axis=0).astype(np.float32))
        self.scalar_bank = torch.from_numpy(std_scalar_bank)
        self.hist_bank = torch.from_numpy(hist_arr.astype(np.float32))
        self.n_min = int(scalar_arr[:, 0].min())
        self.n_max = int(scalar_arr[:, 0].max())
        self.m_min = int(scalar_arr[:, 1].min())
        self.m_max = int(scalar_arr[:, 1].max())
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def scalar_dim(self) -> int:
        return len(self.SCALAR_KEYS)

    @property
    def profile_stat_dim(self) -> int:
        """Width of the standardized profile vector handed to the encoder."""
        return self.scalar_dim + self.n_bins

    # -------------------------------------------------------- (de)standardize
    def standardize_scalar(self, raw_scalar: torch.Tensor) -> torch.Tensor:
        """Standardize a (..., 8) raw scalar tensor with train-core mean/std."""
        if not self._fitted:
            raise RuntimeError("DatasetProfile not fitted.")
        mean = self.scalar_mean.to(raw_scalar.device)
        std = self.scalar_std.to(raw_scalar.device)
        return (raw_scalar - mean) / std

    def destandardize_scalar(self, std_scalar: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("DatasetProfile not fitted.")
        mean = self.scalar_mean.to(std_scalar.device)
        std = self.scalar_std.to(std_scalar.device)
        return std_scalar * std + mean

    def profile_vector(self, raw_scalar: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """Concatenate standardized scalars and raw histogram → encoder-facing vector."""
        return torch.cat([self.standardize_scalar(raw_scalar), hist.to(raw_scalar.device)], dim=-1)

    # ----------------------------------------------------------- sample bank
    def sample_joint(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample (standardized_scalars, histogram) pairs jointly from the train-core bank.

        Preserves correlations between node count, edge count, density, and degree shape.
        """
        if not self._fitted:
            raise RuntimeError("DatasetProfile not fitted.")
        n = self.scalar_bank.shape[0]
        if generator is not None:
            idx = torch.randint(0, n, (batch_size,), generator=generator)
        else:
            idx = torch.randint(0, n, (batch_size,))
        std_scalars = self.scalar_bank[idx].to(device)
        hists = self.hist_bank[idx].to(device)
        return std_scalars, hists

    def aligned_bank(self, adjacencies: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode adjacencies in their current order with this fitted profile.

        ``Datasets.shuffle`` changes the order used by the final condition-code
        collection.  The banks stored during ``fit`` retain the original order,
        so constrained benchmark sampling must rebuild the scalar/histogram bank
        alongside the final condition-code order instead of assuming indices
        still match.
        """
        if not self._fitted:
            raise RuntimeError("DatasetProfile not fitted.")
        scalars = []
        histograms = []
        for adjacency in adjacencies:
            dense = _to_dense_array(adjacency)
            scalars.append(_scalar_stats_from_dense(dense))
            histograms.append(_degree_histogram(dense, self.n_bins))
        if not scalars:
            raise ValueError("DatasetProfile.aligned_bank got an empty adjacency list.")
        raw_scalars = torch.from_numpy(np.stack(scalars).astype(np.float32))
        hist_bank = torch.from_numpy(np.stack(histograms).astype(np.float32))
        return self.standardize_scalar(raw_scalars), hist_bank

    # -------------------------------------------------------------- scoring
    def score_graph(self, g: nx.Graph) -> float:
        """Distance from a candidate graph to the train-core profile mean.

        Weights (per Plan.md Intervention 8):
            node_count    1.0
            edge_count    1.0
            density       1.0
            avg_degree    0.5
            max_degree    0.5
            deg histogram 0.25 (heavily downweighted; already supervised in training)

        deg_moment_2/3/4 are not used in the score (they are correlated with
        avg/max degree and noisier; standardized via DatasetProfile.fit only
        so they are available for future analyses).
        """
        if not self._fitted:
            raise RuntimeError("DatasetProfile not fitted.")
        scalar, hist = _stats_from_graph(g, self.n_bins)
        std = self.scalar_std.numpy()
        mean = self.scalar_mean.numpy()
        z = (scalar - mean) / std  # already standardized so mean-target == 0

        weights = np.array([1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
        scalar_part = float(np.sum(weights * np.abs(z)))
        hist_part = 0.25 * float(np.sum(np.abs(hist - self.hist_mean.numpy())))
        return scalar_part + hist_part

    # -------------------------------------------------------- (de)serialize
    def state_dict(self) -> dict:
        if not self._fitted:
            raise RuntimeError("DatasetProfile not fitted.")
        return {
            "n_bins": self.n_bins,
            "scalar_mean": self.scalar_mean.clone(),
            "scalar_std": self.scalar_std.clone(),
            "hist_mean": self.hist_mean.clone(),
            "scalar_bank": self.scalar_bank.clone(),
            "hist_bank": self.hist_bank.clone(),
            "n_min": self.n_min,
            "n_max": self.n_max,
            "m_min": self.m_min,
            "m_max": self.m_max,
        }

    def load_state_dict(self, state: dict) -> "DatasetProfile":
        self.n_bins = int(state["n_bins"])
        self.scalar_mean = state["scalar_mean"].clone()
        self.scalar_std = state["scalar_std"].clone()
        self.hist_mean = state["hist_mean"].clone()
        self.scalar_bank = state["scalar_bank"].clone()
        self.hist_bank = state["hist_bank"].clone()
        self.n_min = int(state["n_min"])
        self.n_max = int(state["n_max"])
        self.m_min = int(state["m_min"])
        self.m_max = int(state["m_max"])
        self._fitted = True
        return self


# ---------------------------------------------------------------- per-dataset recipes
# Numbers are starting points (Plan.md Intervention 9). Hypsearch sweeps narrowly
# around these. CLI flags override them.
