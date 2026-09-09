import random
import warnings
import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import torch

from flow_klein.data.common import BFS, _compute_structural_node_features, _ensure_csr_adj, compute_graph_stats_from_adj, data_split, grid
from flow_klein.data.benchmarks_v0901 import is_benchmark_dataset, load_benchmark_splits

class Datasets():
  'Characterizes a dataset for PyTorch'
  def __init__(
      self,
      list_adjs,
      self_for_none,
      list_Xs,
      graphlabels = None,
      padding =True,
      Max_num = None,
      set_diag_of_isol_Zer=True,
      node_feat_mode='legacy',
      lap_pe_dim=8
  ):
        """
        :param list_adjs: a list of adjacency in sparse format
        :param list_Xs: a list of node feature matrix
        :param graphlabels: a list of int, that indicate correponding class of element in list_adjs

        """
        'Initialization'
        if Max_num!=0 and Max_num!=None:
            list_adjs, graphlabels, list_Xs = self.remove_largergraphs( list_adjs, graphlabels, list_Xs, Max_num)
        self.set_diag_of_isol_Zer = set_diag_of_isol_Zer
        self.paading = padding
        self.list_Xs = list_Xs
        self.labels = graphlabels
        self.list_adjs = []
        for adj in list_adjs:
            self.list_adjs.append(_ensure_csr_adj(adj))
        self.node_feat_mode = node_feat_mode
        self.lap_pe_dim = lap_pe_dim
        self.toatl_num_of_edges = 0
        self.max_num_nodes = 0
        for i, adj in enumerate(self.list_adjs):
            self.list_adjs[i] = _ensure_csr_adj(
                adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            )
            self.list_adjs[i] = _ensure_csr_adj(
                self.list_adjs[i] + sp.eye(self.list_adjs[i].shape[0], format='csr')
            )
            if self.max_num_nodes < adj.shape[0]:
                self.max_num_nodes = adj.shape[0]
            self.toatl_num_of_edges += adj.sum().item()
            # if list_Xs!=None:
            #     self.list_adjs[i], list_Xs[i] = self.permute(list_adjs[i], list_Xs[i])
            # else:
            #     self.list_adjs[i], _ = self.permute(list_adjs[i], None)
        if Max_num!=None:
            self.max_num_nodes = Max_num
        self.processed_Xs = []
        self.processed_adjs = []
        self.num_of_edges = []
        # for i in range(len(self.list_Xs)):
        for i in range(self.__len__()):
            a,x,n,_ = self.process(i,self_for_none)
            self.processed_Xs.append(x)
            self.processed_adjs.append(a)
            self.num_of_edges.append(n)

        # Check if processed_Xs is not empty before accessing it
        if len(self.processed_Xs) > 0:
            self.feature_size = self.processed_Xs[0].shape[-1]
        else:
            self.feature_size = 0  # Default value when no graphs are processed
        self.adj_s= []
        self.x_s = []
        self.num_nodes = []
        self.subgraph_indexes = []
        self.graph_stats = []

        self.featureList = None

  def remove_largergraphs(self, adjs, labels, Xs, max_size):
      processed_adjs = []
      # Preserve the absence of optional labels/features. Returning [] for a
      # missing label list makes shuffle() treat the dataset as labelled and
      # then index an empty list with every graph index.
      processed_labels = [] if labels is not None else None
      processed_Xs = [] if Xs is not None else None

      for i in range(len(adjs)):
          if adjs[i].shape[0]<=max_size:
              processed_adjs.append(adjs[i])
              if labels is not None:
                  processed_labels.append(labels[i])
              if Xs is not None:
                  processed_Xs.append(Xs[i])
      return processed_adjs,processed_labels,processed_Xs

  def get(self):
      indexces = list(range(self.__len__()))
      return [self.processed_adjs[i] for i in indexces], [self.processed_Xs[i] for i in indexces]

  def set_features(self, some_feature, ):
      self.featureList = some_feature
      # self.labels = labels


  def get_adj_list(self):
      return self.adj_s

  def get__(self,from_, to_, self_for_none, bfs=None, ignore_isolate_nodes = False):
      adj_s = []
      x_s = []
      num_nodes = []
      subgraph_indexes = []
      # padded_to = max([self.list_adjs[i].shape[1] for i in range(from_, to_)])
      # padded_to = 225
      if bfs==None:
          graphfeatures = []
          for element in self.featureList:
              graphfeatures.append(element[from_:to_])
          return (
              self.adj_s[from_:to_],
              self.x_s[from_:to_],
              self.num_nodes[from_:to_],
              self.subgraph_indexes[from_:to_],
              graphfeatures,
              self.graph_stats[from_:to_]
          )

      for i in range(from_, to_):
          # bfs = self.max_num_nodes
          adj, x, num_node, indexes = self.process(i, self_for_none,None, bfs, ignore_isolate_nodes)#, padded_to)
          adj_s.append(adj)
          x_s.append(x)
          num_nodes.append(num_node)
          subgraph_indexes.append(indexes)

      return adj_s, x_s, num_nodes, subgraph_indexes


  def get_max_degree(self):
      return np.max([adj.sum(-1) for adj in self.processed_adjs])
  def processALL(self, self_for_none, bfs=None, ignore_isolate_nodes = False):
      self.adj_s = []
      self.x_s = []
      self.num_nodes = []
      self.subgraph_indexes = []
      self.graph_stats = []
      # padded_to = max([self.list_adjs[i].shape[1] for i in range(from_, to_)])
      # padded_to = 225

      from_ = 0
      to_ = len(self.list_adjs)
      for i in range(from_, to_):
          # bfs = self.max_num_nodes
          adj, x, num_node, indexes = self.process(i, self_for_none,None, bfs, ignore_isolate_nodes)#, padded_to)
          self.adj_s.append(adj)
          self.x_s.append(x)
          self.num_nodes.append(num_node)
          self.subgraph_indexes.append(indexes)
          self.graph_stats.append(
              torch.tensor(
                  compute_graph_stats_from_adj(self.list_adjs[i]),
                  dtype=torch.float32
              )
          )

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_adjs)

  def process(self,index,self_for_none, padded_to=None, bfs_max_length = None, ignore_isolate_nodes=True):
      # self.featureList = None

      if bfs_max_length!=None:
        bfs_max_length = min(bfs_max_length, self.max_num_nodes)

      num_nodes = self.list_adjs[index].shape[0]
      if self.paading == True:
          max_num_nodes = self.max_num_nodes if padded_to==None else padded_to
      else:
          max_num_nodes = num_nodes
      adj_padded = lil_matrix((max_num_nodes,max_num_nodes)) # make the size equal to maximum graph
      if max_num_nodes==num_nodes:
          adj_padded = lil_matrix(self.list_adjs[index], dtype=np.int8)
      else:
        adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
      # adj_padded -= sp.dia_matrix((adj_padded.diagonal()[np.newaxis, :], [0]), shape=adj_padded.shape)
      adj_padded.setdiag(0)
      nodeDegree = adj_padded.sum(-1)
      if not ignore_isolate_nodes:
          nodeDegree+=1

      if self_for_none:
          adj_padded.setdiag(1)
      else:
          if max_num_nodes != num_nodes:
              adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
          else:
              adj_padded += sp.eye(num_nodes)
      # adj_padded+= sp.eye(max_num_nodes)



      if self.node_feat_mode == 'struct':
          X = _compute_structural_node_features(
              self.list_adjs[index],
              self.list_Xs[index] if self.list_Xs is not None else None,
              max_num_nodes=max_num_nodes,
              lap_pe_dim=self.lap_pe_dim
          )
      elif type(self.list_Xs[index]) != np.ndarray:
          # if the feature is not exist we use identical matrix
          diag = np.ones(max_num_nodes)
          if (self.set_diag_of_isol_Zer==True):
            diag[num_nodes:]=0
          X = np.identity( max_num_nodes)
          np.fill_diagonal(X, diag)

          featureVec = np.array(adj_padded.sum(1)) / max_num_nodes
          X= np.concatenate([X,featureVec], 1)
      else:
          #ToDo: deal with data with diffrent number of nodes
          X = self.list_Xs[index]

      # adj_padded, X = self.permute(adj_padded, X)

      # # Converting sparse matrix to sparse tensor
      # coo = adj_padded.tocoo()
      # values = coo.data
      # indices = np.vstack((coo.row, coo.col))
      # i = torch.LongTensor(indices)
      # v = torch.FloatTensor(values)
      # shape = coo.shape
      # adj_padded = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
      X = torch.tensor(X).float()
      # adj_padded, X = permute([adj_padded], [X])
      # adj_padded = adj_padded[0]
      # X = X[0]
      bfs_indexes =set()
      if bfs_max_length!=None:
          while(len(bfs_indexes)<bfs_max_length):
              indexes = set(range(adj_padded.shape[0])).difference(bfs_indexes).difference(np.where(nodeDegree==0)[0])

              source_indx = list(indexes)[np.random.randint(len(indexes))]
              bfs_index = scipy.sparse.csgraph.breadth_first_order(adj_padded, source_indx)
              portionSize = min(len(bfs_index[0]),int(bfs_max_length/5))
              if (portionSize+len(bfs_indexes)>=bfs_max_length):
                  bfs_indexes =bfs_indexes.union(bfs_index[0][:(bfs_max_length-len(bfs_indexes))])
              else:
                  bfs_indexes = bfs_indexes.union(bfs_index[0][:portionSize])
          bfs_indexes = list(bfs_indexes)

      if len(bfs_indexes)==0:
          bfs_indexes = list(range(max_num_nodes))

          # nodeDegree = adj_padded.sum(-1)
          # indexes = set(range(adj_padded.shape[0]))
          # non_isolate_nodes = list(set(range(adj_padded.shape[0])).difference(np.where(nodeDegree<2)[0]))
          # source_indx = non_isolate_nodes[np.random.randint(len(non_isolate_nodes))]
          # bfs_indexes = scipy.sparse.csgraph.breadth_first_order(adj_padded, source_indx)
          # bfs_indexes = np.concatenate((bfs_indexes[0],np.where(nodeDegree<2)[0]))
      # none_selected = set(range(adj_padded.shape[0])).difference(set(bfs_indexes))

      # index = bfs_indexes + list(none_selected)
      # adj_padded= adj_padded[:,index]
      # adj_padded = adj_padded[index, :]
      # X = X[:,index]
      # X = X[index, :]

      return adj_padded, X, num_nodes,bfs_indexes
  # def process(self,index,self_for_none, padded_to=None,):
  #
  #     num_nodes = self.list_adjs[index].shape[0]
  #     if self.paading == True:
  #         max_num_nodes = self.max_num_nodes if padded_to==None else padded_to
  #     else:
  #         max_num_nodes = num_nodes
  #     adj_padded = lil_matrix((max_num_nodes,max_num_nodes)) # make the size equal to maximum graph
  #     if max_num_nodes==num_nodes:
  #         adj_padded = lil_matrix(self.list_adjs[index], dtype=np.int8)
  #     else:
  #       adj_padded[:num_nodes, :num_nodes] = self.list_adjs[index][:, :]
  #     adj_padded -= sp.dia_matrix((adj_padded.diagonal()[np.newaxis, :], [0]), shape=adj_padded.shape)
  #     if self_for_none:
  #       adj_padded += sp.eye(max_num_nodes)
  #     else:
  #         if max_num_nodes != num_nodes:
  #             adj_padded[:num_nodes, :num_nodes] += sp.eye(num_nodes)
  #         else:
  #             adj_padded += sp.eye(num_nodes)
  #     # adj_padded+= sp.eye(max_num_nodes)
  #
  #
  #
  #
  #     if self.list_Xs == None:
  #         # if the feature is not exist we use identical matrix
  #         X = np.identity( max_num_nodes)
  #         node_degree = adj_padded.sum(0)
  #         X = np.concatenate((node_degree.transpose(), X),1 )
  #
  #     else:
  #         #ToDo: deal with data with diffrent number of nodes
  #         X = self.list_Xs[index].toarray()
  #
  #     # adj_padded, X = self.permute(adj_padded, X)
  #
  #     # Converting sparse matrix to sparse tensor
  #     coo = adj_padded.tocoo()
  #     values = coo.data
  #     indices = np.vstack((coo.row, coo.col))
  #     i = torch.LongTensor(indices)
  #     v = torch.FloatTensor(values)
  #     shape = coo.shape
  #     adj_padded = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
  #     X = torch.tensor(X, dtype=torch.int8)
  #
  #     return adj_padded.reshape(1,*adj_padded.shape), X.reshape(1, *X.shape), num_nodes

  # def permute(self, list_adj, X):
  #           p = list(range(list_adj.shape[0]))
  #           np.random.shuffle(p)
  #           # for i in range(list_adj.shape[0]):
  #           #     list_adj[:, i] = list_adj[p, i]
  #           #     X[:, i] = X[p, i]
  #           # for i in range(list_adj.shape[0]):
  #           #     list_adj[i, :] = list_adj[i, p]
  #           #     X[i, :] = X[i, p]
  #           list_adj[:, :] = list_adj[p, :]
  #           list_adj[:, :] = list_adj[:, p]
  #           if X !=None:
  #               X[:, :] = X[p, :]
  #               X[:, :] = X[:, p]
  #           return list_adj , X

  def shuffle(self):
      indx = list(range(len(self.list_adjs)))
      np.random.shuffle(indx)

      if self.list_Xs is not None:
        if len(self.list_Xs) != len(self.list_adjs):
          raise ValueError(
              "Node features and adjacency lists must have the same length: "
              f"features={len(self.list_Xs)}, graphs={len(self.list_adjs)}"
          )
        self.list_Xs=[self.list_Xs[i] for i in indx]
      else:
          warnings.warn("X is empty")

      self.list_adjs=[self.list_adjs[i] for i in indx]

      # if the graphs have extracted features
      if self.featureList !=None:
          for el_i , element in enumerate(self.featureList):
              self.featureList[el_i] = element[indx]
      else:
          warnings.warn("Graph structureal feature is an empty Set")

      if self.labels is not None:
          if len(self.labels) != len(self.list_adjs):
              raise ValueError(
                  "Labels and adjacency lists must have the same length: "
                  f"labels={len(self.labels)}, graphs={len(self.list_adjs)}"
              )
          self.labels= [self.labels[i] for i in indx]
      else:
           warnings.warn("Label is an empty Set")

      if len(self.subgraph_indexes)>0:
          self.adj_s= [self.adj_s[i] for i in indx]
          self.x_s = [self.x_s[i] for i in indx]
          self.num_nodes = [self.num_nodes[i] for i in indx]
          self.subgraph_indexes = [self.subgraph_indexes[i] for i in indx]
          self.graph_stats = [self.graph_stats[i] for i in indx]


  def __getitem__(self, index):
        'Generates one sample of data'
        # return self.processed_adjs[index], self.processed_Xs[index],torch.tensor(self.list_adjs[index].todense(), dtype=torch.float32)
        return self.processed_adjs[index], self.processed_Xs[index]


def structural_BFS(list_adj):
    """Deterministically order nodes by structure, then stable BFS traversal.

    This is intentionally separate from the historical ``BFS`` function so
    Grid and all legacy datasets keep their exact preprocessing path.  Roots
    and neighbours are ranked by degree, core number, and closeness; the
    original integer node id is used only as a deterministic final tie-break.
    """
    for graph_index, adjacency in enumerate(list_adj):
        adjacency = _ensure_csr_adj(adjacency)
        node_count = adjacency.shape[0]
        if node_count <= 1:
            list_adj[graph_index] = adjacency
            continue

        if hasattr(nx, "from_scipy_sparse_array"):
            graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.Graph)
        else:
            graph = nx.from_scipy_sparse_matrix(adjacency, create_using=nx.Graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        degrees = dict(graph.degree())
        try:
            cores = nx.core_number(graph) if graph.number_of_edges() else {
                node: 0 for node in graph.nodes()
            }
        except nx.NetworkXError:
            cores = {node: 0 for node in graph.nodes()}
        closeness = nx.closeness_centrality(graph)

        def structural_key(node):
            return (
                int(degrees.get(node, 0)),
                int(cores.get(node, 0)),
                float(closeness.get(node, 0.0)),
                -int(node),
            )

        unseen = set(graph.nodes())
        order = []
        while unseen:
            root = max(unseen, key=structural_key)
            queue = [root]
            unseen.remove(root)
            cursor = 0
            while cursor < len(queue):
                node = queue[cursor]
                cursor += 1
                order.append(node)
                neighbours = [nbr for nbr in graph.neighbors(node) if nbr in unseen]
                neighbours.sort(key=structural_key, reverse=True)
                for neighbour in neighbours:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        queue.append(neighbour)

        list_adj[graph_index] = _ensure_csr_adj(adjacency[order, :][:, order])
    return list_adj

def list_graph_loader(graph_type, _max_list_size=None, return_labels=False,
                      limited_to=None, shuffle=True, shuffle_seed=None):
  list_adj = []
  list_x = []
  list_labels = []
  if is_benchmark_dataset(graph_type):
      benchmark_splits = load_benchmark_splits(graph_type)
      for split in ("train", "val", "test"):
          list_adj.extend(benchmark_splits[split])
      list_x = [None for _ in list_adj]
  elif graph_type=="grid":
      for i in range(10, 20):
        for j in range(10, 20):
            list_adj.append(nx.adjacency_matrix(grid(i, j)))
            list_x.append(None)
  else:
      raise ValueError("Unsupported dataset for v0901: " + str(graph_type))

  def return_subset(A,X,Y, limited_to):
      indx = list(range(len(A)))
      if shuffle:
          subset_rng = random if shuffle_seed is None else random.Random(shuffle_seed)
          subset_rng.shuffle(indx)
      A = [_ensure_csr_adj(A[i]) for i in indx]
      X = [X[i] for i in indx]
      if Y!=None and len(Y)!=0 : Y = [Y[i] for i in indx]

      if limited_to != None:

          A = A[:limited_to]
          X = X[:limited_to]
          if Y!=None and len(Y)!=0 : Y = Y[:limited_to]
      return A,X,Y

  if return_labels ==True:
      if len(list_labels)==0:
          list_labels = None
  return return_subset(list_adj, list_x, list_labels, limited_to)

__all__ = ['Datasets', 'BFS', 'data_split', 'list_graph_loader', 'structural_BFS']
