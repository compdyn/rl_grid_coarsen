import numpy as np
import torch as T
from torch.nn import ReLU, GRU, Sequential, Linear

from torch import sigmoid, softmax, relu, tanh

from collections import namedtuple, deque
from torch_geometric.nn import TAGConv
import random
import scipy as sp
import time
import Batch_Graph as bg
from itertools import islice 
import time
from DuelingNet import Net as Net_TAGConv


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
    
    def __init__(self, dim, K, gamma, epsilon, lr, mem_size, batch_size, eps_min=0.01,
                 eps_dec=1e-4, replace=10):
        
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
        self.memory = ReplayBuffer(mem_size)
        self.learn_step_counter = 0
     
        self.q_eval = Net_TAGConv(self.dim, self.K, self.lr)
        
        self.q_targ = Net_TAGConv(self.dim, self.K, self.lr)
        
        
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
        
        self.q_eval.optimizer.zero_grad()
        
        
        self.replace_target_network()
        
        states, list_viols, num_viols, actions, rewards, next_states,\
            next_list_viols, next_num_viols, nodes, masks = self.memory.sample(self.batch_size)
        
        loss = 0

        b_states = bg.Batch.from_data_list(states)
        b_next_states = bg.Batch.from_data_list(next_states)
        
        
        Q_prime = self.q_targ.forward(b_next_states, b_next_states)[0].detach()
        
        
        Qmodd = self.q_eval.forward(b_next_states, b_next_states)[0].detach()
        
        
        Qmodd = Qmodd.flatten().tolist()
        
        Inputt = iter(Qmodd) 
        splited_Qmodd = [list(islice(Inputt, elem)) for elem in nodes] 
    
        Qprime = T.zeros(self.batch_size)
        Q      = T.zeros(self.batch_size) 
        argmax = []
        
        
        Q1 = self.q_eval.forward(b_states, b_states)[0]
        
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
            
            
        Qprime.flatten()
        
        
        y = T.tensor(rewards) + self.gamma*Qprime*T.tensor(masks)
        
    
        loss = self.q_eval.loss(Q,y)

            
        self.loss = loss
                
        loss.backward()
                
        self.q_eval.optimizer.step()  
                
        self.learn_step_counter += 1
        

        
        
        
        
        
        