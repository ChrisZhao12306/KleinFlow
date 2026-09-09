import random
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import torch

def _safe_dense_feature_array(raw_x, num_nodes):
    """Convert optional node features to a dense numpy array."""
    if raw_x is None:
        return None
    if torch.is_tensor(raw_x):
        raw_x = raw_x.detach().cpu().numpy()
    elif sp.issparse(raw_x):
        raw_x = raw_x.toarray()
    else:
        raw_x = np.asarray(raw_x)

    if raw_x.ndim == 1:
        raw_x = raw_x[:, None]
    if raw_x.shape[0] != num_nodes:
        return None
    return raw_x.astype(np.float32, copy=False)


def _normalized_laplacian_csr(adj_matrix):
    """Build a normalized Laplacian without scipy's sparse getA1 path."""
    adj_matrix = _ensure_csr_adj(adj_matrix).astype(np.float32)
    degrees = np.asarray(adj_matrix.sum(axis=1)).reshape(-1).astype(np.float32)
    isolated_mask = degrees == 0

    inv_sqrt_degree = np.zeros_like(degrees, dtype=np.float32)
    nonzero_mask = ~isolated_mask
    inv_sqrt_degree[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])

    degree_scale = sp.diags(inv_sqrt_degree, offsets=0, format='csr')
    normalized_adj = degree_scale @ adj_matrix @ degree_scale

    laplacian = sp.eye(adj_matrix.shape[0], format='csr', dtype=np.float32) - normalized_adj
    if isolated_mask.any():
        laplacian = laplacian.tolil()
        laplacian[isolated_mask, isolated_mask] = 0.0
        laplacian = laplacian.tocsr()
    return laplacian


def _compute_structural_node_features(adj_matrix, raw_x, max_num_nodes, lap_pe_dim=8):
    """
    Build fixed-dimensional structural node features for graph generation.

    Features:
    - normalized degree
    - log(1 + degree)
    - clustering coefficient
    - core number
    - PageRank
    - valid-node mask
    - Laplacian positional encodings
    - optional raw node features
    """
    if isinstance(adj_matrix, sp.spmatrix):
        adj_matrix = adj_matrix.tocsr().astype(np.float32)
    elif sp.issparse(adj_matrix):
        # NetworkX 3.x returns scipy sparse arrays; convert them back to the
        # csr_matrix API required by the batching utilities.
        adj_matrix = sp.csr_matrix(adj_matrix, dtype=np.float32)
    else:
        adj_matrix = sp.csr_matrix(np.asarray(adj_matrix, dtype=np.float32))

    adj_matrix = adj_matrix.copy()
    adj_matrix = adj_matrix - sp.dia_matrix((adj_matrix.diagonal()[np.newaxis, :], [0]), shape=adj_matrix.shape)
    adj_matrix.eliminate_zeros()

    num_nodes = adj_matrix.shape[0]
    if hasattr(nx, "from_scipy_sparse_array"):
        graph_nx = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.Graph)
    else:
        graph_nx = nx.from_scipy_sparse_matrix(adj_matrix, create_using=nx.Graph)
    degrees = np.asarray(adj_matrix.sum(1)).reshape(-1).astype(np.float32)

    degree_denom = max(float(max(num_nodes - 1, 1)), 1.0)
    norm_degree = degrees / degree_denom
    log_degree = np.log1p(degrees)

    if graph_nx.number_of_nodes() > 0 and graph_nx.number_of_edges() > 0:
        clustering = np.array(
            [value for _, value in nx.clustering(graph_nx).items()],
            dtype=np.float32
        )
        core_numbers = np.array(
            [value for _, value in nx.core_number(graph_nx).items()],
            dtype=np.float32
        )
        pagerank = np.array(
            [value for _, value in nx.pagerank(graph_nx).items()],
            dtype=np.float32
        )
    else:
        clustering = np.zeros(num_nodes, dtype=np.float32)
        core_numbers = np.zeros(num_nodes, dtype=np.float32)
        pagerank = np.full(num_nodes, 1.0 / max(num_nodes, 1), dtype=np.float32)

    core_scale = max(float(core_numbers.max()), 1.0)
    core_numbers = core_numbers / core_scale

    lap_dim = min(lap_pe_dim, max(num_nodes - 1, 0))
    lap_pe = np.zeros((num_nodes, lap_pe_dim), dtype=np.float32)
    if lap_dim > 0:
        laplacian = _normalized_laplacian_csr(adj_matrix)
        try:
            eigen_count = min(num_nodes - 1, lap_dim + 1)
            _, eigenvectors = eigsh(
                laplacian.asfptype(),
                k=eigen_count,
                which='SM'
            )
            usable = eigenvectors[:, 1:lap_dim + 1] if eigenvectors.shape[1] > 1 else eigenvectors[:, :lap_dim]
            lap_pe[:, :usable.shape[1]] = usable.astype(np.float32)
        except Exception:
            pass

    valid_mask = np.ones((num_nodes, 1), dtype=np.float32)
    struct_features = np.concatenate(
        [
            norm_degree[:, None],
            log_degree[:, None],
            clustering[:, None],
            core_numbers[:, None],
            pagerank[:, None],
            valid_mask,
            lap_pe,
        ],
        axis=1
    )

    raw_x = _safe_dense_feature_array(raw_x, num_nodes)
    if raw_x is not None:
        struct_features = np.concatenate([raw_x, struct_features], axis=1)

    padded = np.zeros((max_num_nodes, struct_features.shape[1]), dtype=np.float32)
    padded[:num_nodes] = struct_features
    if max_num_nodes > num_nodes:
        padded[num_nodes:, -1 - lap_pe_dim] = 0.0
    return padded


def compute_graph_stats_from_adj(adj_matrix):
    """Return graph-level structural targets used by the auxiliary prediction heads."""
    if not sp.issparse(adj_matrix):
        dense_adj = np.asarray(adj_matrix, dtype=np.float32)
    else:
        dense_adj = adj_matrix.toarray()
    dense_adj = dense_adj.astype(np.float32, copy=False)
    np.fill_diagonal(dense_adj, 0.0)

    num_nodes = float(dense_adj.shape[0])
    edge_count = float(dense_adj.sum() / 2.0)
    max_edges = max(num_nodes * max(num_nodes - 1.0, 0.0) / 2.0, 1.0)
    density = edge_count / max_edges
    degrees = dense_adj.sum(axis=1)
    avg_degree = float(degrees.mean()) if degrees.size > 0 else 0.0
    degree_sq_mean = float(np.mean(np.square(degrees))) if degrees.size > 0 else 0.0
    degree_cube_mean = float(np.mean(np.power(degrees, 3))) if degrees.size > 0 else 0.0
    return np.array(
        [num_nodes, edge_count, density, avg_degree, degree_sq_mean, degree_cube_mean],
        dtype=np.float32
    )


def _ensure_csr_adj(adj):
    """Normalize adjacency containers to scipy CSR matrices."""
    if isinstance(adj, sp.spmatrix):
        return adj.tocsr()
    if sp.issparse(adj):
        return sp.csr_matrix(adj)
    return sp.csr_matrix(np.asarray(adj))


def data_split(graph_lis, list_x=None, list_label=None, split_seed=123):
    # Shuffle with a local RNG so the fixed benchmark split does not overwrite
    # the training/generation RNG selected by --seed.
    split_rng = random.Random(split_seed)
    index = list(range(len(graph_lis)))
    split_rng.shuffle(index)
    graph_lis = [graph_lis[i] for i in index]

    if list_x!=None:
        list_x = [list_x[i] for i in index]

    if list_label!=None:
        list_label = [list_label[i] for i in index]

    #----------------------------------------

    graph_test_len = len(graph_lis)

    graph_train = graph_lis[0:int(0.8 * graph_test_len)]  # train
    # graph_validate = graph_lis[0:int(0.2 * graph_test_len)]  # validate
    graph_test = graph_lis[int(0.8 * graph_test_len):]  # test on a hold out test set

    list_x_train = list_x_test = None
    if list_x!=None:
        list_x_train = list_x[0:int(0.8 * graph_test_len)]  # train
        list_x_test = list_x[int(0.8 * graph_test_len):]

    list_label_train = list_label_test = None
    if list_label!=None:
        list_label_train = list_label[0:int(0.8 * graph_test_len)]  # train
        list_label_test = list_label[int(0.8 * graph_test_len):]

    return  graph_train, graph_test, list_x_train , list_x_test,list_label_train, list_label_test


def BFS(list_adj):
    for i, adj in enumerate(list_adj):
        adj = _ensure_csr_adj(adj)
        num_nodes = adj.shape[0]
        if num_nodes <= 1:
            list_adj[i] = adj
            continue

        remaining_nodes = set(range(num_nodes))
        bfs_order = []

        # Preserve all nodes by running BFS over each disconnected component.
        while remaining_nodes:
            source = min(remaining_nodes)
            bfs_index, _ = sp.csgraph.breadth_first_order(
                adj, source, directed=False, return_predecessors=True
            )
            visited = [node for node in bfs_index.tolist() if node in remaining_nodes]
            if len(visited) == 0:
                visited = [source]

            bfs_order.extend(visited)
            remaining_nodes.difference_update(visited)

        list_adj[i] = _ensure_csr_adj(adj[bfs_order, :][:, bfs_order])
    return list_adj

def grid(m= 10, n=10 ):
    # https: // networkx.github.io / documentation / stable / auto_examples / drawing / plot_four_grids.html
    G = nx.grid_2d_graph(m, n)  # 4x4 grid
    return G
