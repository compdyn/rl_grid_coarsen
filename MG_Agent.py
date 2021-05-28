import numpy as np
import torch as T
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
#from NeuralNet import Net
from DuelingNet import Net as Net_TAGConv
from DuelingNet2 import Net as Net_MPNN
from FCNet import Net as Net_FC


class ReplayBuffer():
    
    def __init__(self, max_size):
        
        self.mem_size = max_size
        self.replay_buffer = deque(maxlen=max_size)
        
    def store(self, state, list_viol, num_viol, action, reward, next_state\
                       , next_list_viol, next_num_viol, node, mask):
        
        experience = namedtuple('Experience', ['state', 'list_viol', 'num_viol',\
                    'action', 'reward','next_state', 'next_list_viol', 'next_num_viol',\
                        'node','mask'])
            
        e = experience(state, list_viol, num_viol, action, reward, next_state\
                       , next_list_viol, next_num_viol, node, mask)
            
        self.replay_buffer.append(e)
        
    def sample(self, batch_size):
        
        experiences = random.sample(self.replay_buffer, k=batch_size)
        
        states          = [e.state for e in experiences if e is not None]    
        list_viols      = [e.list_viol for e in experiences if e is not None]
        num_viols       = [e.num_viol for e in experiences if e is not None]
        actions         = [e.action for e in experiences if e is not None]
        rewards         = [e.reward for e in experiences if e is not None]
        next_states     = [e.next_state for e in experiences if e is not None]
        next_list_viols = [e.next_list_viol for e in experiences if e is not None]
        next_num_viols  = [e.next_num_viol for e in experiences if e is not None]
        nodes           = [e.node for e in experiences if e is not None]
        masks           = [e.mask for e in experiences if e is not None]
        
        return (states, list_viols, num_viols, actions, rewards, \
                next_states, next_list_viols, next_num_viols, nodes, masks)
        

class Agent():
    
    def __init__(self, dim, K, gamma, epsilon, lr, mem_size, batch_size, net_type, eps_min=0.01,
                 eps_dec=1e-4, replace=10): #, chkpt_dir='tmp/unstructured_gnn'):
        
        #self.num_nodes = num_nodes
        self.gamma = gamma 
        self.epsilon = epsilon
        self.lr = lr
        self.dim = dim
        self.loss = T.tensor([0])
        self.K = K
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_targe_cnt = replace
        #self.chkpt_dir = chkpt_dir
        self.memory = ReplayBuffer(mem_size)
        self.learn_step_counter = 0
        self.net_type = net_type
        
        if self.net_type == 'FC':
            self.q_eval = Net_FC(self.dim, self.K, self.lr)#, self.num_nodes) #, name='unstructured_q_eval',
                              #chkpt_dir=self.chkpt_dir)
            
            self.q_targ = Net_FC(self.dim, self.K, self.lr)#, self.num_nodes) #, name='unstructured_q_targ',
                              #chkpt_dir=self.chkpt_dir)
                              
        if self.net_type == 'MPNN':
            self.q_eval = Net_MPNN(self.dim, self.K, self.lr)#, self.num_nodes) #, name='unstructured_q_eval',
                              #chkpt_dir=self.chkpt_dir)
            
            self.q_targ = Net_MPNN(self.dim, self.K, self.lr)#, self.num_nodes) #, name='unstructured_q_targ',
                              #chkpt_dir=self.chkpt_dir)
                              
        if self.net_type == 'TAGConv':
            self.q_eval = Net_TAGConv(self.dim, self.K, self.lr)#, self.num_nodes) #, name='unstructured_q_eval',
                              #chkpt_dir=self.chkpt_dir)
            
            self.q_targ = Net_TAGConv(self.dim, self.K, self.lr)#, self.num_nodes) #, name='unstructured_q_targ',
                              #chkpt_dir=self.chkpt_dir)
        
        
    def choose_action(self, state, viol_nodes):
        
        
            
        if np.random.random()> self.epsilon:
            
            with T.no_grad():
                
                advantage = self.q_eval.forward(state)[0]
                action = viol_nodes[T.argmax(advantage[viol_nodes]).item()]
          
        else:
            
            action = np.random.choice(viol_nodes)
        
        return action
    
    def store_transition(self, state, list_viols, num_viol, \
                         action, reward, next_state, next_list_viol, next_num_viol, node, mask):
        
        self.memory.store(state, list_viols, num_viol, \
                         action, reward, next_state, next_list_viol, next_num_viol, node, mask)
        
    def replace_target_network(self):
        
        if self.learn_step_counter % self.replace_targe_cnt == 0:
            self.q_targ.load_state_dict(self.q_eval.state_dict())
            
            
    def decrement_epsilon(self):

        self.epsilon = self.epsilon - self.eps_dec\
            if self.epsilon>self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_targ.save_checkpoint()
        
    def load_models(self):
        self.q_eval.load_checkpoint()   
        self.q_targ.load_checkpoint()
        
    def learn(self):
        
        if len(self.memory.replay_buffer) < self.batch_size:
            return
        
        # t1 = time.time()
        self.q_eval.optimizer.zero_grad()
        
        
        self.replace_target_network()
        
        states, list_viols, num_viols, actions, rewards, next_states,\
            next_list_viols, next_num_viols, nodes, masks = self.memory.sample(self.batch_size)
        
        loss = 0
        '''
        for i in range(self.batch_size):
            
            
            Q_prime = self.q_targ.forward(next_states[i])[0].detach()
        
            Qmodd = self.q_eval.forward(next_states[i])[0].detach()#.squeeze(0)
            
            if len(next_list_viols[i]) != 0:
                
                argmax = np.array(next_list_viols[i][Qmodd[next_list_viols[i]].argmax(0)])
                
            else:
                
                argmax = 0
                
            argmax = T.tensor(argmax).unsqueeze(0)
        
            # print ("argmax", argmax)
            Qprime = Q_prime.gather(0,T.tensor(argmax).unsqueeze(1).long())
            Qprime = Qprime.flatten()
            
            y = T.tensor(rewards[i]) + self.gamma*Qprime*T.tensor(masks[i])
            
            Q1 = self.q_eval.forward(states[i])[0]#.squeeze(0)
            Q = Q1.gather(0, T.tensor(actions[i]).unsqueeze(0).unsqueeze(0).long())
            Q = Q.squeeze(1)
            
            
            loss += self.q_eval.loss(Q,y)
            
            # if len(next_list_viols[i]) == 0:
            #     print (masks[i])
            #     print (loss)
            #     print (Q)
            #     print (y)
            #     print ("*********")
            
        
        self.loss = loss
        loss.backward()
        self.q_eval.optimizer.step()  
                
    
        self.q_eval.optimizer.zero_grad()
            
        #self.decrement_epsilon()
    
                
            
            # if loss>5:
            #     print (list_viols[i])
            #     print (next_list_viols[i])
            #     print (actions[i])
            #     print ("*********")
            
        
        
        '''
        b_states = bg.Batch.from_data_list(states)
        b_next_states = bg.Batch.from_data_list(next_states)
        
        # t2 = time.time()
        
        Q_prime = self.q_targ.forward(b_next_states, b_next_states)[0].detach()
        
        # t3 = time.time()
        
        Qmodd = self.q_eval.forward(b_next_states, b_next_states)[0].detach()#.squeeze(0)
        
        # t4 = time.time()
        
        Qmodd = Qmodd.flatten().tolist()
        
        Inputt = iter(Qmodd) 
        splited_Qmodd = [list(islice(Inputt, elem)) for elem in nodes] 
    
        Qprime = T.zeros(self.batch_size)
        Q      = T.zeros(self.batch_size) 
        argmax = []
        
        # t5 = time.time()
        
        Q1 = self.q_eval.forward(b_states, b_states)[0]#.squeeze(0)
        
        # t6 = time.time()
        idx_in_batch = 0
        for i in range(self.batch_size):
            
            if i>0:
                idx_in_batch += nodes[i-1]
                actions = (np.array(actions) + nodes[i-1]).tolist()
            
            if (np.array(splited_Qmodd[i])[next_list_viols[i]]).tolist() != []:
                
                argmax = np.array(next_list_viols[i])[np.argmax(\
                                        np.array(splited_Qmodd[i])[next_list_viols[i]])]
                    
                argmax = argmax+idx_in_batch
              
                Qprime[i] = Q_prime.gather(0, T.tensor(argmax).unsqueeze(0).unsqueeze(0).long())
                            
            else:
                
                argmax = 0
            
            Q[i]      = Q1.gather(0, T.tensor(actions[i]).unsqueeze(0).unsqueeze(0).long())
            #print (Q_prime.shape)
            
            
        Qprime.flatten()
        
        
        y = T.tensor(rewards) + self.gamma*Qprime*T.tensor(masks)
        
    
        loss = self.q_eval.loss(Q,y)
        # if loss>100:
            
        #     print (Q)
        #     print (y)
            
        #     kiri = T.argmin(Q)
        #     print(states[kiri].x)
        #     print(states[kiri].edge_attr)
            
        self.loss = loss
        
        # t7 = time.time()
        
        loss.backward()
        
        # t8 = time.time()
        
        self.q_eval.optimizer.step()  
        
        # t9 = time.time()
        
        self.learn_step_counter += 1
        
        
     
        # print ("T21", t2-t1)
        # print ("T32", t3-t2)
        # print ("T43", t4-t3)
        # print ("T54", t5-t4)
        # print ("T65", t6-t5)
        # print ("T76", t7-t6)
        # print ("T87", t8-t7)
        # print ("T98", t9-t8)
        
        
        
        
        
        