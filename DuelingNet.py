import numpy as np
import torch as T
import torch
import os
import torch.optim as optim
from torch.nn import ReLU, GRU, Sequential, Linear
import torch.nn.functional as F
import torch.nn as nn
from torch import sigmoid, softmax, relu, tanh
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple, deque
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, GATConv, graclus, max_pool, max_pool_x, 
                                global_mean_pool, BatchNorm, InstanceNorm, GraphConv,
                                GCNConv, TAGConv, SGConv, LEConv, TransformerConv, SplineConv,
                                GMMConv, GatedGraphConv, ARMAConv, GENConv, DeepGCNLayer,
                                LayerNorm)
import random
import scipy as sp
import time
import Batch_Graph as bg
from itertools import islice 
import time

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_mean
#from torch_geometric.nn import MetaLayer

'''
class EdgeModel(torch.nn.Module):
    def __init__(self,dim_in, dim, dim_out):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(dim_in, dim), ReLU(), Lin(dim, dim_out))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self,dim_in1,dim_in2, dim, dim_out):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(dim_in1, dim), ReLU(), Lin(dim, dim))
        self.node_mlp_2 = Seq(Lin(dim_in2, dim), ReLU(), Lin(dim, dim_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, dim):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(dim, dim), ReLU(), Lin(dim, dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)
    
    
    
class Net(torch.nn.Module):
    
    def __init__(self, dim, K, lr, num_nodes):
        
        super(Net, self).__init__()
        
        
        
        self.m1  = MetaLayer(EdgeModel(5, dim, dim), NodeModel(dim+2,dim+2, dim, dim))#, GlobalModel(dim))
        self.m2  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        # self.m3  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        # self.m4  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        # self.m5  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        # self.m6  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))#, GlobalModel(dim))
        # self.m7  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        # self.m8  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        # self.m9  = MetaLayer(EdgeModel(3*dim, dim, dim), NodeModel(2*dim,2*dim, dim, dim))
        self.m10 = MetaLayer(EdgeModel(3*dim, dim, 1), NodeModel(1+dim,2*dim, dim, 1))
        
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)
        
    def forward(self, D):
        
        x = D.x
        edge_index = D.edge_index
        edge_attr = D.edge_attr
        
        x, edge_attr,_ = self.m1(x, edge_index, edge_attr)
        x = relu(x)
        edge_attr = relu(edge_attr)
    
        x, edge_attr,_ = self.m2(x, edge_index, edge_attr)
        x = F.elu(x)
        edge_attr = F.elu(edge_attr)
        # x, edge_attr,_ = self.m3(x, edge_index, edge_attr)
        # x = relu(x)
        # edge_attr = relu(edge_attr)
        # x, edge_attr,_ = self.m4(x, edge_index, edge_attr)
        # x = F.elu(x)
        # edge_attr = F.elu(edge_attr)
        # x, edge_attr,_ = self.m5(x, edge_index, edge_attr)
        # x = relu(x)
        # edge_attr = relu(edge_attr)
        # x, edge_attr,_ = self.m6(x, edge_index, edge_attr)
        # x = F.elu(x)
        # edge_attr = F.elu(edge_attr)
        # x, edge_attr,_ = self.m7(x, edge_index, edge_attr)
        # x = relu(x)
        # edge_attr = relu(edge_attr)
        # x, edge_attr,_ = self.m8(x, edge_index, edge_attr)
        # x = F.elu(x)
        # edge_attr = F.elu(edge_attr)
        # x, edge_attr,_ = self.m9(x, edge_index, edge_attr)
        # x = relu(x)
        # edge_attr = relu(edge_attr)
        x, edge_attr,_ = self.m10(x, edge_index, edge_attr)
        
        return x
        
from typing import Optional, Tuple

import torch
from torch import Tensor


class MetaLayer(torch.nn.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    A graph network takes a graph as input and returns an updated graph as
    output (with same connectivity).
    The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
    as well as global-level features :obj:`u`.
    The output graph has the same structure, but updated features.

    Edge features, node features as well as global features are updated by
    calling the modules :obj:`edge_model`, :obj:`node_model` and
    :obj:`global_model`, respectively.

    To allow for batch-wise graph processing, all callable functions take an
    additional argument :obj:`batch`, which determines the assignment of
    edges or nodes to their specific graphs.

    Args:
        edge_model (Module, optional): A callable which updates a graph's edge
            features based on its source and target node features, its current
            edge features and its global features. (default: :obj:`None`)
        node_model (Module, optional): A callable which updates a graph's node
            features based on its current node features, its graph
            connectivity, its edge features and its global features.
            (default: :obj:`None`)
        global_model (Module, optional): A callable which updates a graph's
            global features based on its node features, its graph connectivity,
            its edge features and its current global features.

    .. code-block:: python

        from torch.nn import Sequential as Seq, Linear as Lin, ReLU
        from torch_scatter import scatter_mean
        from torch_geometric.nn import MetaLayer

        class EdgeModel(torch.nn.Module):
            def __init__(self):
                super(EdgeModel, self).__init__()
                self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, src, dest, edge_attr, u, batch):
                # source, target: [E, F_x], where E is the number of edges.
                # edge_attr: [E, F_e]
                # u: [B, F_u], where B is the number of graphs.
                # batch: [E] with max entry B - 1.
                out = torch.cat([src, dest, edge_attr, u[batch]], 1)
                return self.edge_mlp(out)

        class NodeModel(torch.nn.Module):
            def __init__(self):
                super(NodeModel, self).__init__()
                self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
                self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                row, col = edge_index
                out = torch.cat([x[row], edge_attr], dim=1)
                out = self.node_mlp_1(out)
                out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
                out = torch.cat([x, out, u[batch]], dim=1)
                return self.node_mlp_2(out)

        class GlobalModel(torch.nn.Module):
            def __init__(self):
                super(GlobalModel, self).__init__()
                self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

            def forward(self, x, edge_index, edge_attr, u, batch):
                # x: [N, F_x], where N is the number of nodes.
                # edge_index: [2, E] with max entry N - 1.
                # edge_attr: [E, F_e]
                # u: [B, F_u]
                # batch: [N] with max entry B - 1.
                out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
                return self.global_mlp(out)

        op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
        x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super(MetaLayer, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        

        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(
            self, x, edge_index, edge_attr, u: Optional[Tensor] = None,
            batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """"""
        # x = D.x
        # edge_index = D.edge_index
        # edge_attr = D.edge_attr

        # print (x.shape)
        # print (edge_attr.shape)
        # print (edge_index.shape)
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u,
                                        batch if batch is None else batch[row])

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, batch)

        if self.global_model is not None:
            u = self.global_model(x, edge_index, edge_attr, u, batch)
            
        return x, edge_attr, u


    def __repr__(self):
        return ('{}(\n'
                '    edge_model={},\n'
                '    node_model={},\n'
                '    global_model={}\n'
                ')').format(self.__class__.__name__, self.edge_model,
                            self.node_model, self.global_model)


# x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)

'''

class Net(T.nn.Module):

    def __init__(self, dim, K, lr):#, name,chkpt_dir):
        
        super(Net, self).__init__()
       
        
        self.conv1 = TAGConv(2, dim, K = K)
        self.conv2 = TAGConv(dim, dim, K = K)
        self.conv_A = TAGConv(dim, 1, K = K)
        self.conv_V = TAGConv(dim, 1, K=K)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)
        

    def forward(self, D, b_states = None):
        
   
        x, edge_index, edge_attr = D.x, D.edge_index, D.edge_attr
        # x = self.lin0(x)                                            #comment for TAG conv
        # edge_attr = self.linattr(edge_attr)                         #comment for TAG conv
        
        edge_attr = edge_attr.flatten()                            #uncomment for TAG conv
        data = relu(self.conv1(x, edge_index, edge_attr))
        data = relu(self.conv2(data, edge_index, edge_attr))
        #edge_attr = self.linattr(D.edge_attr)
        # data_A = 0.5*tanh(self.conv_A(data, edge_index, edge_attr))-0.5
        data_A = self.conv_A(data, edge_index, edge_attr)
        #data_A = self.agg_A(data_A)                                    #comment for TAG conv
        
        data_V = self.conv_V(data, edge_index, edge_attr)
        
        #data_V = self.agg_V(data_V)                                    #comment for TAG conv
        
        if b_states != None:
            list_idxs = [torch.nonzero(b_states.batch == i).flatten().tolist() for i in range(len(b_states.ptr)-1)]
    
            V_s = torch.ones((b_states.x.shape[0],1))
            max_A_s = torch.ones((b_states.x.shape[0],1))
            
            for i in range(len(b_states.ptr)-1):
                
                
                V_s[list_idxs[i]] = data_V[list_idxs[i]].mean()
                max_A_s[list_idxs[i]] = data_A[list_idxs[i]].max()
          
            data   = V_s + data_A - max_A_s
            
        else:   
            data_V = data_V.mean()
            data   = data_V+data_A - data_A.max()

        return data, data_A
        

