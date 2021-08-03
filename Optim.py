from  Unstructured import MyMesh, rand_Amesh_gen, rand_grid_gen, grid
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy
import fem
import networkx as nx
import numpy as np
import scipy as sp
import pygmsh
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random
import torch as T
from torch_geometric.data import Data
import Batch_Graph as bg
import copy
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
import torch_geometric
from torch_geometric.data import Data
from pyamg.gallery import poisson
import matplotlib as mpl
import os
from MG_Agent import Agent
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc
from pyamg import amg_core
from pyamg.graph import lloyd_cluster
from Scott_greedy import greedy_coarsening

import sys

# list(list(G.edges(data=True))[1][-1].values())


def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = T.from_numpy(A.row).to(T.long)
    col = T.from_numpy(A.col).to(T.long)
    edge_index = T.stack([row, col], dim=0)
    edge_weight = T.from_numpy(A.data)
    return edge_index, edge_weight


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = T.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = T.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data

        
        
def structured(n_row, n_col, Theta):
    
    num_nodes = int(n_row*n_col)

    X = np.array([[i/(n_col*n_row) for i in range(n_col)] for j in range(n_row)]).flatten()
    Y = np.array([[j/(n_row*n_col) for i in range(n_col)] for j in range(n_row)]).flatten()
    E = []
    V = []
    nv = num_nodes
    N = [i for i in range(num_nodes)]
    
    epsilon = 1
    theta = 1 #param of A matrix
   
    sten = diffusion_stencil_2d(epsilon=epsilon,theta=theta,type='FD')
    AA = stencil_grid(sten, (n_row, n_col), dtype=float, format='csr')
    
    nz_row = []
    nz_col = []
    t1 = time.time()
       
    for i in range(n_row):
        for j in range(n_col):
            
            if i!=n_row-1:
                if j!=n_col-1:
                    
                    nz_row.append(i*n_col+j)
                    nz_row.append(i*n_col+j)
                    nz_col.append(i*n_col+j+1)
                    nz_col.append(i*n_col+j+n_col)
                else:
                    nz_row.append(i*n_col+j)
                    nz_col.append(i*n_col+j+n_col)
                    
            if i == n_row-1:
                if j!=n_col-1:
                    
                    nz_row.append(i*n_col+j)
                    nz_col.append(i*n_col+j+1)
                    
                    
    nz_row = np.array(nz_row)
    nz_col = np.array(nz_col)
    
    
    # print ("t21", t2-t1)
    e = np.concatenate((np.expand_dims(nz_row,axis=1), np.expand_dims(nz_col, axis=1)), axis=1)
    Edges = list(tuple(map(tuple, e)))
    num_edges = len(Edges)
    g = rand_grid_gen(None)
    
    mesh = copy.deepcopy(g.mesh)
    
    mesh.X = X
    mesh.Y = Y
    mesh.E = E
    mesh.V = V
    mesh.nv = nv
    mesh.ne = []
    mesh.N = N
    mesh.Edges = Edges
    mesh.num_edges = num_edges
    
    fine_nodes = [i for i in range(num_nodes)]

    grid_ = grid(AA,fine_nodes,[], mesh, Theta)

    
    # print ("t21", t2-t1)
    # print ("t32", t3-t2)
    # print ("t43", t4-t3)
    return grid_

def lloyd_aggregation(C, ratio=0.03, distance='unit', maxiter=10):
    """Aggregate nodes using Lloyd Clustering.

    Parameters
    ----------
    C : csr_matrix
        strength of connection matrix
    ratio : scalar
        Fraction of the nodes which will be seeds.
    distance : ['unit','abs','inv',None]
        Distance assigned to each edge of the graph G used in Lloyd clustering

        For each nonzero value C[i,j]:

        =======  ===========================
        'unit'   G[i,j] = 1
        'abs'    G[i,j] = abs(C[i,j])
        'inv'    G[i,j] = 1.0/abs(C[i,j])
        'same'   G[i,j] = C[i,j]
        'sub'    G[i,j] = C[i,j] - min(C)
        =======  ===========================

    maxiter : int
        Maximum number of iterations to perform

    Returns
    -------
    AggOp : csr_matrix
        aggregation operator which determines the sparsity pattern
        of the tentative prolongator
    seeds : array
        array of Cpts, i.e., Cpts[i] = root node of aggregate i

    See Also
    --------
    amg_core.standard_aggregation

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation.aggregate import lloyd_aggregation
    >>> A = poisson((4,), format='csr')   # 1D mesh with 4 vertices
    >>> A.todense()
    matrix([[ 2., -1.,  0.,  0.],
            [-1.,  2., -1.,  0.],
            [ 0., -1.,  2., -1.],
            [ 0.,  0., -1.,  2.]])
    >>> lloyd_aggregation(A)[0].todense() # one aggregate
    matrix([[1],
            [1],
            [1],
            [1]], dtype=int8)
    >>> # more seeding for two aggregates
    >>> Agg = lloyd_aggregation(A,ratio=0.5)[0].todense()

    """
    if ratio <= 0 or ratio > 1:
        raise ValueError('ratio must be > 0.0 and <= 1.0')

    if not (isspmatrix_csr(C) or isspmatrix_csc(C)):
        raise TypeError('expected csr_matrix or csc_matrix')

    if distance == 'unit':
        data = np.ones_like(C.data).astype(float)
    elif distance == 'abs':
        data = abs(C.data)
    elif distance == 'inv':
        data = 1.0/abs(C.data)
    elif distance is 'same':
        data = C.data
    elif distance is 'min':
        data = C.data - C.data.min()
    else:
        raise ValueError('unrecognized value distance=%s' % distance)

    if C.dtype == complex:
        data = np.real(data)

    assert(data.min() >= 0)

    G = C.__class__((data, C.indices, C.indptr), shape=C.shape)

    num_seeds = int(min(max(ratio * G.shape[0], 1), G.shape[0]))

    distances, clusters, seeds = lloyd_cluster(G, num_seeds, maxiter=maxiter)

    row = (clusters >= 0).nonzero()[0]
    col = clusters[row]
    data = np.ones(len(row), dtype='int8')
    AggOp = coo_matrix((data, (row, col)),
                       shape=(G.shape[0], num_seeds)).tocsr()
    
    return AggOp, seeds, col



sz_list = [100*(i+1) for i in range(1)]
K = 4
agent = Agent(dim = 32, K = K, gamma = 1, epsilon = 1, \
                      lr= 0.001, mem_size = 5000, net_type = 'TAGConv', batch_size = 64,  \
                          eps_min = 0.01 , eps_dec = 1.333/5000, replace=10)

agent.q_eval.load_state_dict(T.load('Models/MPNN/Dueling_MPNN900.pth'))
#agent.q_eval.load_state_dict(T.load('Models/Dueling_batch_train_final.pth'))
agent.epsilon = 0
list_size = []
list_time  = []


def Post_processing(num_iter, agent, grid_, K):
    
    for _ in range(num_iter):
        
        ffrac = sum(grid_.active)/grid_.num_nodes
        copy_grid = copy.deepcopy(grid_)
        center = np.random.randint(0,grid_.num_nodes)
            
        region2 = grid_.node_hop_neigh(center, 2*K)
        region  = grid_.node_hop_neigh(center, K)
        
        indices = []
        newly_added = []
        for node in region:
            
            news = grid_.uncoarsen(node)
            newly_added.append(news)
            indices.append(region2.index(node))
            
        done = False
        
        while not done:
            
            data    = grid_.subgrid(region2)
            Q, advantage = agent.q_eval.forward(data)
            
            viols_idx = grid_.is_violating[region2].nonzero()[0].tolist()
            viols  = np.array(region2)[viols_idx].tolist()
            
            if len(viols_idx) != 0:
                
                node_max = viols[T.argmax(advantage[viols_idx])]
                    
                newly_ = grid_.coarsen_node(node_max)
            
            done = True if len(viols_idx) == 0 else False
        
        if ffrac > sum(grid_.active)/grid_.num_nodes:
        
            grid_ = copy_grid
        
            
    
    grid_.fine_nodes = grid_.active.nonzero()[0].tolist()#list(set(grid_.fine_nodes)-set(maxes))
    grid_.coarse_nodes = np.nonzero(grid_.active == 1)[0].tolist()
    grid_.violating_nodes = grid_.is_violating.nonzero()[0].tolist()
    
    ffrac = sum(grid_.active)/grid_.num_nodes
    
    return grid_, ffrac
    
    

def Linear_Coarsening_Lloyd(g_, agent, Greedy):
    
    grid_ = copy.deepcopy(g_)
    grid_ = grid(grid_.A, grid_.fine_nodes, grid_.coarse_nodes, grid_.mesh, grid_.Theta)

    if not Greedy:
        
        observation = grid_.data
        
        with T.no_grad():
            Q, advantage = agent.q_eval.forward(observation)
        
        adv_tensor = copy.deepcopy(advantage)
        
        done = False
        
        _,_,index_agg = lloyd_aggregation(grid_.A,ratio=0.033,maxiter=1000)
        
        list_agg = []
        num_aggs = index_agg.max()+1
        for i in range(num_aggs):
            
            list_agg.append(np.nonzero(index_agg==i)[0].tolist())
            
        while not done:
            
            viols = grid_.violating_nodes
            
            for idx in range(num_aggs):
                
                aggreg = np.array(list_agg [idx])
                viols  = aggreg[grid_.is_violating[aggreg.tolist()].nonzero()[0]].tolist()
                
                if len(viols) != 0:
                    node_max = viols[T.argmax(adv_tensor[viols])]
                    
                    _ = grid_.coarsen_node(node_max)
    
            observation = grid_.data
            # grid_.active[maxes] = 0
            # grid_.is_violating[newly_removed] = 0
            grid_.fine_nodes = grid_.active.nonzero()[0].tolist()#list(set(grid_.fine_nodes)-set(maxes))
            grid_.violating_nodes = grid_.is_violating.nonzero()[0].tolist()
    
            Q, adv_tensor = agent.q_eval.forward(observation)
      
    
            done = True if len(grid_.violating_nodes) == 0 else False

    else:
        
        grid_ = greedy_coarsening(grid_)
        
    return grid_
    
