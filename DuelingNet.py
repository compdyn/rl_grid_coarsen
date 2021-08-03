import numpy as np
import torch as T
import torch
import torch.optim as optim
from torch.nn import ReLU, GRU, Sequential, Linear
import torch.nn as nn
from torch import sigmoid, relu

from torch_geometric.nn import TAGConv


class Net(T.nn.Module):

    def __init__(self, dim, K, lr):
        
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
     
        
        edge_attr = edge_attr.flatten()                            
        data = relu(self.conv1(x, edge_index, edge_attr))
        data = relu(self.conv2(data, edge_index, edge_attr))

        data_A = self.conv_A(data, edge_index, edge_attr)
                                    
        
        data_V = self.conv_V(data, edge_index, edge_attr)
        
        
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
        

