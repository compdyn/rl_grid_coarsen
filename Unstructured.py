import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import scipy
import fem
import scipy as sp
import pygmsh
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import random
import torch as T
import torch_geometric
import Batch_Graph as bg
import copy
import networkx as nx
from networkx.drawing.nx_pylab import draw_networkx
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from torch_geometric.data import Data
from pyamg.aggregation import lloyd_aggregation
from pyamg.gallery import poisson
import matplotlib as mpl
import os
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_csc
from pyamg import amg_core
from pyamg.graph import lloyd_cluster
import sys
from MG_Agent import Agent

mpl.rcParams['figure.dpi'] = 300


class MyMesh:
    def __init__(self, mesh):
        
        self.X = mesh.points[:,0:1].flatten()
        self.Y = mesh.points[:,1:2].flatten()
        self.E = mesh.cells[1].data
        self.V = mesh.points[:,0:2]
        self.nv = mesh.points[:,0:2].shape[0]
        self.ne = len(mesh.cells[1].data)
        
        e01 = self.E[:,[0,1]]
        e02 = self.E[:,[0,2]]
        e12 = self.E[:,[1,2]]
    
        e01 = tuple(map(tuple, e01))
        e02 = tuple(map(tuple, e02))
        e12 = tuple(map(tuple, e12))
        
        e = list(set(e01).union(set(e02)).union(set(e12)))
        self.N = [i for i in range(self.X.shape[0])]
        self.Edges = e
        self.num_edges = len(e)
        
      
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
    A = AA.toarray()
    
    nz_row = np.nonzero(A)[0]
    nz_col = np.nonzero(A)[1]
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
    
    return grid_
    
# class grid:
    
#     def __init__(self, A, fine_nodes, coarse_nodes, mesh, Theta):
        
#         self.A = A.tocsr()
#         self.fine_nodes = fine_nodes
#         self.coarse_nodes = coarse_nodes
#         self.num_nodes = mesh.nv
#         #self.edges = set_edge
#         self.mesh = mesh
#         active = np.ones(self.num_nodes)
#         active[self.coarse_nodes] = 0
#         self.active = active
#         self.Theta = Theta
#         AA = abs(self.A)
#         self.AA = AA
    
#         colsz  = self.num_nodes
        
#         A_flat = self.AA.reshape(self.num_nodes**2, 1).tocsr()
#         edges_list = A_flat.nonzero()[0]
#         edge_row  = edges_list//colsz
#         edge_row  = edge_row.tolist()
#         edge_col  = edges_list% colsz
#         edge_col  = edge_col.tolist()
        
#         edge_attr = T.tensor(A_flat.data)
#         edge_attr = edge_attr.unsqueeze(1).float()
#         edge_index = T.tensor([edge_row, edge_col],dtype=T.long)
#         self.edge_index = edge_index
#         self.edge_attr = edge_attr
        
  
#         self.x = T.cat((T.from_numpy(self.active).unsqueeze(1), \
#                         T.from_numpy(self.active).unsqueeze(1)),dim=1).float()
        
        

#         list_neighbours = [[] for i in range(self.x.shape[0])]
#         for i in range(self.edge_index.shape[1]):

#             list_neighbours[self.edge_index[0,i].clone().tolist()].append(self.edge_index[1\
#                                                                   ,i].clone().tolist()) 
#         self.list_neighbours = list_neighbours 
        
#         deg_l = []
#         for l in self.list_neighbours:
#             deg_l.append(len(l))
#         deg_arr = np.array(deg_l)
        
#         self.deg_std = deg_arr.std()
#         self.deg_mean = deg_arr.mean()
    
#         self.data = Data(x=self.x, edge_index=edge_index, edge_attr= edge_attr)
        
#         self.violating_nodes = self.viol_nodes()[0]
#         self.is_violating = self.viol_nodes()[1]
        
#     def subgrid(self, node_list):
        
#         sub_x = self.x[node_list]
#         '''
#         n = len(node_list)
#         sub_AA = np.zeros((n,n))
#         for i in range(300):
#             for j in range(300):
#                 if i<n and j<n:
#                     sub_AA[i,j] = self.AA[node_list[i], node_list[j]]
#                 else:
#                     sub_AA[0,0] = self.AA[node_list[0], node_list[0]]
                    
#         sub_AA = scipy.sparse.csr_matrix(sub_AA)
#         '''
#         sub_AA = self.AA[node_list,:]
#         sub_AA = sub_AA[:,node_list]
#         colsz  = sub_AA.shape[1]
        
#         A_flat = sub_AA.reshape(sub_AA.shape[0]*sub_AA.shape[1], 1).tocsr()
#         edges_list = A_flat.nonzero()[0]
#         edge_row  = edges_list//colsz
#         edge_row  = edge_row.tolist()
#         edge_col  = edges_list% colsz
#         edge_col  = edge_col.tolist()
        
#         sub_edge_attr = T.tensor(A_flat.data)
#         sub_edge_attr = sub_edge_attr.unsqueeze(1).float()
#         sub_edge_index = T.tensor([edge_row, edge_col],dtype=T.long)
              
#         sub_data = Data(x=sub_x, edge_index=sub_edge_index, edge_attr= sub_edge_attr)
        
#         return sub_data
    
        
#     def is_viol(self, node):
        
#         if self.active[node] == 0:
#             return False
        
#         else:
            
#             ##newly added###
            
#             # for neigh in self.list_neighbours[node]:
#             #     if self.active[neigh] != 0:
#             #         if self.AA[neigh,neigh]<(self.Theta*self.AA[neigh,:]*np.expand_dims(self.active,1))[0,0]:
#             #             return True
                    
#             ##end##
#             if self.AA[node,node]>(self.Theta*self.AA[node,:]*np.expand_dims(self.active,1))[0,0]:
#                 return False
#             else:
#                 return True
        
#     def viol_nodes(self):
        
#         active_nodes = self.active
#         AA = self.AA
#         I = scipy.sparse.csr_matrix(np.identity(self.num_nodes))
#         active_aij = AA.multiply(active_nodes).tocsr()
#         sum_aij = active_aij.sum(1)
#         B = AA.multiply(I)
#         Diag = B.diagonal()
#         vals = np.sign(Diag-self.Theta*np.array(sum_aij).flatten())
#         viols = np.nonzero(vals == -1)[0].tolist()
        
#         ###newly added###
        
#         # viols2 = []
#         # for node in viols:
#         #     viols2.append(node)
#         #     for neighbors in self.list_neighbours[node]:
#         #         viols2.append(neighbors)
#         # viols = viols2
        
#         ###end###
        
#         violatings = list(set(viols).difference(self.coarse_nodes))
#         isviol = T.zeros(self.num_nodes)
#         isviol[violatings] = 1
#         num_viol = len(violatings)
        
#         return violatings, isviol, num_viol
        
  
#     def coarsen_node(self, node_a):
        
    
#         self.fine_nodes.remove(node_a)
#         self.coarse_nodes.append(node_a)
#         self.active[node_a] = 0
#         self.violating_nodes.remove(node_a)
#         self.is_violating[node_a] = 0
        
#         for neigh in self.list_neighbours[node_a]:
#             if self.is_viol(neigh) == False and neigh in self.violating_nodes:
#                 self.violating_nodes.remove(neigh)
#                 self.is_violating[neigh] = 0
        
#         self.x = T.cat((T.from_numpy(self.active).unsqueeze(1), \
#                     self.is_violating.unsqueeze(1).double()),dim=1).float()

#         self.data = Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        
#     def plot(self):
        
#         G = nx.Graph()
#         mymsh = self.mesh
        
#         points = mymsh.N
#         edges  = mymsh.Edges
        
#         pos_dict = {}
#         for i in range(mymsh.nv):
#             pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]
            
#         G.add_nodes_from(points)
#         G.add_edges_from(edges)
#         colors = [i for i in range(mymsh.nv)]
        
#         for i in self.fine_nodes:
#             colors[i] = 'b'
#         for i in self.coarse_nodes:
#             colors[i] = 'r'
#         for i in self.viol_nodes()[0]:
#             colors[i] = 'g'
        
#         draw_networkx(G, pos=pos_dict, with_labels=False, node_size=6, \
#                       node_color = colors, node_shape = 'o')
#         '''
#         plt.plot(points[self.fine_nodes,0], points[self.fine_nodes,1], 'b.',\
#                   points[self.coarse_nodes,0], points[self.coarse_nodes,1], 'r.',\
#                   points[self.viol_nodes()[0],0], points[self.viol_nodes()[0],1], 'g.')
#         '''
        

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



class grid:
    
    def __init__(self, A, fine_nodes, coarse_nodes, mesh, Theta):

        self.A = A.tocsr()
        self.fine_nodes = fine_nodes
        self.coarse_nodes = coarse_nodes
        self.num_nodes = mesh.nv
        #self.edges = set_edge
        self.mesh = mesh
        active = np.ones(self.num_nodes)
        active[self.coarse_nodes] = 0
        self.active = active
        self.Theta = Theta
        
        self.G = nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False)

  
        self.x = T.cat((T.from_numpy(self.active).unsqueeze(1), \
                        T.from_numpy(self.active).unsqueeze(1)),dim=1).float()

        
        edge_index, edge_attr = from_scipy_sparse_matrix(abs(self.A))
        
        list_neighbours1 = []
        list_neighbours2 = []
        for node in range(self.num_nodes):
            a =  list(self.G.edges(node,data = True))
            l1 = []
            l2 = []
            for i in range(len(a)):
                l1.append(a[i][1])
                l2.append(abs(np.array(list(a[i][-1].values())))[0])
                
            list_neighbours1.append(l1)
            list_neighbours2.append(l2)
                
        self.list_neighbours = [list_neighbours1, list_neighbours2]
        
        self.data = Data(x=self.x, edge_index=edge_index, edge_attr= edge_attr.float())
        self.violating_nodes = [i for i in range(self.num_nodes)]               #self.viol_nodes()[0]
        self.is_violating = np.array([1 for i in range(self.num_nodes)])        #self.viol_nodes()[1]

        
    def subgrid(self, node_list):

        sub_x = self.x[node_list]
        sub_data = from_networkx(self.G.subgraph(node_list))
        sub_data = Data(x=sub_x, edge_index=sub_data.edge_index, edge_attr= abs(sub_data.weight.float()))
        
        return sub_data
    
        
    def node_hop_neigh(self, node, K):
        
        return list(nx.single_source_shortest_path(self.G, node, cutoff=K).keys())
    
    
    def is_viol(self, node):

        if self.active[node] == 0:
            return False
        
        else:

            neigh_list = self.list_neighbours[0][node]#list(self.G.neighbors(node))
            actives    = self.active[neigh_list]
            aij = self.list_neighbours[1][node]
            # aij = np.array([abs(np.array(list(self.G.get_edge_data(node,neigh).values())[0])) \
            #                 for neigh in neigh_list])
            aij = aij*actives
            aij = aij.sum()
            if abs(np.array(list(self.G.get_edge_data(node,node).values())[0]))< self.Theta*aij:
            
                return True
                
            else:

                return False
                
                
    def viol_nodes(self):

        violatings = []
        isviol = []
        
        for node in range(self.num_nodes):
            
            if self.active[node]!=0:
                
                neigh_list = self.list_neighbours[0][node]
                
                #neigh_list = list(self.G.neighbors(node))
                actives    = self.active[neigh_list]
                # aij = np.array([abs(np.array(list(self.G.get_edge_data(node,neigh).values())[0])) \
                #                 for neigh in neigh_list])
                aij = self.list_neighbours[1][node]  
                aij = aij*actives
                aij = aij.sum()
                
                if self.G.get_edge_data(node,node) != None:
                    
                    if abs(np.array(list(self.G.get_edge_data(node,node).values())[0]))< self.Theta*aij:
                    
                        isviol.append(1)
                        violatings.append(node)
                        
                    else:
        
                        isviol.append(0)
    
            else:
                
                isviol.append(0)
        
        num_viol = len(violatings)
        
        return violatings, isviol, num_viol
        
  
    def coarsen_node(self, node_a):
        
        #tkir1  = time.time()
        newly_removed = []
        #self.fine_nodes.remove(node_a)
        self.coarse_nodes.append(node_a)
        self.active[node_a] = 0
        #self.violating_nodes.remove(node_a)
        self.is_violating[node_a] = 0
        newly_removed.append(node_a)
        
        for neigh in self.list_neighbours[0][node_a]:#self.G.neighbors(node_a):
            if self.is_viol(neigh) == False and self.is_violating[neigh] == 1:
                #self.violating_nodes.remove(neigh)
                self.is_violating[neigh] = 0
                newly_removed.append(neigh)

                
                
        
        self.data.x[node_a, 0]        = 0
        self.data.x[newly_removed, 1] = 0
        
        return newly_removed

    def uncoarsen(self, node_a):
        
        self.fine_nodes.append(node_a)
        #self.coarse_nodes.remove(node_a)
        self.active[node_a] = 1
        #self.violating_nodes.remove(node_a)
        #self.is_violating[node_a] = 0
        newly_added = []
        
        if self.is_viol(node_a) == True and self.is_violating[node_a] == 0:
            self.is_violating[node_a] = 1
            newly_added.append(node_a)
    
        
        for neigh in self.list_neighbours[0][node_a]:#self.G.neighbors(node_a):
            if self.is_viol(neigh) == True and self.is_violating[neigh] == 0:
                
                self.is_violating[neigh] = 1
                newly_added.append(neigh)
        
        
        self.data.x[node_a, 0]        = 1
        self.data.x[newly_added, 1]   = 1
        
        return newly_added
    
    
    def plot(self, size, w):
        
        G = nx.Graph()
        mymsh = self.mesh
        
        points = mymsh.N
        edges  = mymsh.Edges
        
        pos_dict = {}
        for i in range(mymsh.nv):
            pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]
            
        G.add_nodes_from(points)
        G.add_edges_from(edges)
        colors = [i for i in range(mymsh.nv)]
        
        for i in self.fine_nodes:
            colors[i] = 'b'
        for i in self.coarse_nodes:
            colors[i] = 'r'
        for i in self.viol_nodes()[0]:
            colors[i] = 'g'
        
        draw_networkx(G, pos=pos_dict, with_labels=False, node_size=size, \
                      node_color = colors, node_shape = 'o', width = w)
        
        plt.axis('equal')

        
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


        
def set_edge_from_msh(msh):
        
    edges = msh.E
    array_of_tuples = map(tuple, edges[:,[1,2]])
    t12 = tuple(array_of_tuples)
    array_of_tuples = map(tuple, edges[:,[0,2]])
    t02 = tuple(array_of_tuples)
    array_of_tuples = map(tuple, edges[:,[0,1]])
    t01 = tuple(array_of_tuples)
    
    set_edge = set(t01).union(set(t02)).union(set(t12))
    
    return set_edge
        

# def rand_Amesh_gen(mesh):
    
#     num_Qhull_nodes = random.randint(45, 90)
#     points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
#     hull = ConvexHull(points)
    
#     msh_sz = 0.15*random.random()+0.25
    
#     if mesh == None:
    
        
#         with pygmsh.geo.Geometry() as geom:
            
#             poly = geom.add_polygon(
                
#                     hull.points[hull.vertices.tolist()].tolist()
                    
#                 ,
#                 mesh_size=msh_sz,
#             )
#             mesh = geom.generate_mesh()
        
#         '''
#             geom.set_mesh_size_callback(
#                 lambda dim, tag, x, y, z: 6.0e-2 + 2.0e-1 * (x ** 2 + y ** 2)
#             )
            
#             field0 = geom.add_boundary_layer(
#                 edges_list=[poly.curves[0], poly.curves[2]],
#                 lcmin=0.05,
#                 lcmax=0.2,
#                 distmin=0.0,
#                 distmax=0.2,
#             )
#             field1 = geom.add_boundary_layer(
#                 nodes_list=[poly.points[8], poly.points[2]],
#                 lcmin=0.05,
#                 lcmax=0.2,
#                 distmin=0.1,
#                 distmax=0.4,
#             )
#             geom.set_background_mesh([field0, field1], operator="Min")
      
#             mesh = geom.generate_mesh()
        
        
#         '''
#         '''
#         with pygmsh.geo.Geometry() as geom:
#             geom.add_polygon(
#                 [
#                     [-1.0, -1.0],
#                     [+1.0, -1.0],
#                     [+1.0, +1.0],
#                     [-1.0, +1.0],
#                 ]
#             )
#             geom.set_mesh_size_callback(
#                 lambda dim, tag, x, y, z: 6.0e-2 + 2.0e-1 * (x ** 2 + y ** 2)
#             )
        
#             mesh = geom.generate_mesh()
            
#         '''
#         '''
#         with pygmsh.geo.Geometry() as geom:
#             poly = geom.add_polygon(
#                 [
                    
                    
#                     [0.0, 0.0],
#                     [0.7, 0.5],
#                     [1.0, 0.0],
                    
                    
                    
#                 ],
#                 mesh_size=0.1
#                 ,
#             )
        
#             field0 = geom.add_boundary_layer(
#                 edges_list=[poly.curves[0], poly.curves[1], poly.curves[2]],
#                 lcmin=0.01,
#                 lcmax=0.1,
#                 distmin=0.15,
#                 distmax=0.01,
#             )
#             field1 = geom.add_boundary_layer(
#                 nodes_list=[poly.points[2]],
#                 lcmin=0.05,
#                 lcmax=0.2,
#                 distmin=0.1,
#                 distmax=0.4,
#             )
#             geom.set_background_mesh([field0], operator="Min")
        
#             mesh = geom.generate_mesh()
#          '''
         
            
#     mymsh = MyMesh(mesh)
#     # points = mymsh.V
#     # tri = Delaunay(points)
#     # plt.triplot(points[:,0], points[:,1], tri.simplices)
#     # plt.plot(points[:,0], points[:,1], 'o')
    
#     A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1)
#     return A, mymsh


def func1(x,y,p):
    
    x_f = int(np.floor(p.shape[0]*x))
    y_f = int(np.floor(p.shape[1]*y))
    
    return p[x_f, y_f]

def rand_Amesh_gen(mesh):
    
    num_Qhull_nodes = random.randint(45, 90)
    points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
    hull = ConvexHull(points)
    
    msh_sz = 0.5 #0.1*random.random()+0.1
    
    
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            
                hull.points[hull.vertices.tolist()].tolist()
                
            ,
            mesh_size=msh_sz,
        )
        
        prob = np.random.random()
        if prob>5:
            
            min_ = 0.005+0.01*np.random.random()
            min_sz  = 0.1#/(min_**0.1)
            p = min_ + min_sz*np.random.random((500,500))
            geom.set_mesh_size_callback(
                #lambda dim, tag, x, y, z: func(x, y, points,min_dist, thresh, min_sz)
                lambda dim, tag, x, y, z: func1(x, y, p)
            )
            
            #geom.set_background_mesh([field0, field1], operator="Min")
        
        mesh = geom.generate_mesh()
    
    
    
    mymsh = MyMesh(mesh)
    # points = mymsh.V
    # tri = Delaunay(points)
    # plt.triplot(points[:,0], points[:,1], tri.simplices)
    # plt.plot(points[:,0], points[:,1], 'o')
    
    A,b = fem.gradgradform(mymsh, kappa=None, f=None, degree=1)
    
    return A, mymsh


#T.save(mesh, "mesh.pth")
#mesh = T.load("mesh.pth")



def rand_grid_gen(mesh):
    
    A, mymsh = rand_Amesh_gen(mesh)
    fine_nodes = [i for i in range(A.shape[0])]
    
    #set_of_edge = set_edge_from_msh(mymsh)
    rand_grid = grid(A,fine_nodes,[],mymsh,0.56)
    
    return rand_grid



def plot_cycle(list_grid, base_node_size, edge_width, scale):
    
    #shapes = ['o', '^', 's']
    node_color_list = ['k', 'k', 'orange', 'yellow' ]
    for index in range(len(list_grid)):
        
        node_color = node_color_list[index]
        
        grid_ = copy.deepcopy(list_grid[index])
        G = nx.Graph()
        mymsh = grid_.mesh
        
        points = mymsh.N
        
        if index == 0:
            edges  = mymsh.Edges
        else:
            edges = []
            
        pos_dict = {}
        for i in range(mymsh.nv):
            pos_dict[i] = [mymsh.X[i], mymsh.Y[i]]
            
        G.add_nodes_from(points)
        G.add_edges_from(edges)
        colors = [i for i in range(mymsh.nv)]
                    
        for i in grid_.fine_nodes:
            colors[i] = node_color
        for i in grid_.coarse_nodes:
            colors[i] = node_color
        for i in grid_.viol_nodes()[0]:
            colors[i] = 'g'
            
          
        
        draw_networkx(G, pos=pos_dict, with_labels=False, node_size = base_node_size*(1*index+scale*index**2), \
                      node_color = colors, edge_color = 'k', \
                          node_shape = 'o', width = edge_width)
        plt.axis('equal')
    
    


import gmsh
import torch
import meshio
#mesh = meshio.read('Test_Graphs/Hand_crafted/Geometry/Graph1.msh')

class gmsh2MyMesh:
    def __init__(self, mesh):
        
        
        diff = set([i for i in range(mesh.points[:,0:2].shape[0])]) - \
            set(mesh.cells[-1].data.flatten().tolist())
            
        mesh.points = np.delete(mesh.points, list(diff), axis=0)
        arr_diff = np.array(list(diff))
        for i in range(len(mesh.cells[-1].data)):
            
            for j in range(3):
                
                shift = mesh.cells[-1].data[i,j]>arr_diff
                shift = np.sum(shift)
                mesh.cells[-1].data[i,j] = mesh.cells[-1].data[i,j] - shift
            
        self.X = mesh.points[:,0:1].flatten()
        self.Y = mesh.points[:,1:2].flatten()
        self.E = mesh.cells[-1].data
        self.V = mesh.points[:,0:2]
        self.nv = mesh.points[:,0:2].shape[0]
        self.ne = len(mesh.cells[-1].data)
        
        e01 = self.E[:,[0,1]]
        e02 = self.E[:,[0,2]]
        e12 = self.E[:,[1,2]]
    
        e01 = tuple(map(tuple, e01))
        e02 = tuple(map(tuple, e02))
        e12 = tuple(map(tuple, e12))
        
        e = list(set(e01).union(set(e02)).union(set(e12)))
        self.N = [i for i in range(self.X.shape[0])]
        self.Edges = e
        self.num_edges = len(e)
    

def hand_grid(mesh):
    
    msh = gmsh2MyMesh(mesh)
    
    A,b = fem.gradgradform(msh, kappa=None, f=None, degree=1)
    
    fine_nodes = [i for i in range(A.shape[0])]
    
    #set_of_edge = set_edge_from_msh(mymsh)
    rand_grid = grid(A,fine_nodes,[],msh,0.56)
    
    return rand_grid




    
    
    
    
    
    