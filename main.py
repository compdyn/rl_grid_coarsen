import numpy as np
from MG_Agent import Agent
#from utils import plot_learning_curves
import Unstructured as uns
import time
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from  Unstructured import MyMesh, grid, rand_Amesh_gen, rand_grid_gen, structured
import fem 
#from torch.utils.tensorboard import SummaryWriter
import sys
import torch as T
from Scott_greedy import greedy_coarsening
import copy

#writer= SummaryWriter("runs")

if __name__ == '__main__':
    
    done = False
    load_ckeckpoint = False
    K = 4
    learning_per_step = 5
    num_steps = 10000
    learn_every = 4
    count = 0
    dim_list = [32]
    lr_list = [0.001]
    loss_list = []
    iterat = 0
    

    for lr in lr_list:
        for dim in dim_list:
            
            agent = Agent(dim = 32, K = K, gamma = 1, epsilon = 1,
                  lr= lr, mem_size = 5000, batch_size = 32,
                      eps_min = 0.01 , eps_dec = 1.25/num_steps, replace=10)
            loss_list = []
            
            
            if load_ckeckpoint:
                
                agent.load_models()
            
            
            for i in range(num_steps):

                done = False
                g_idx = np.random.randint(0,50)
                
                grid_ = rand_grid_gen(None)
                agent.decrement_epsilon()
                while not done:
                    
                    observation = grid_.data
                    list_viols = grid_.viol_nodes()[0]
         
                    action = agent.choose_action(observation, list_viols)
                    grid_.coarsen_node(action)
                    num_c_nodes = len(grid_.coarse_nodes)
                    next_list_viols = grid_.viol_nodes()[0]
                    next_observation = grid_.data
                    #reward = -100/grid_.num_nodes
                    reward = -1#200*num_c_nodes/(grid_.num_nodes**2)
                    done = True if grid_.viol_nodes()[2] == 0 else False
                    agent.store_transition(observation, list_viols,\
                                          None, action, reward,\
                                          next_observation, next_list_viols,\
                                          None, grid_.num_nodes, 1-int(done))
                        
                    if count % learn_every == 0:
                        
                        for __ in range(learning_per_step):
                            agent.learn()
                            loss = agent.loss.item()
                            loss = loss*agent.memory.mem_size/len(agent.memory.replay_buffer)
                            loss_list.append(loss)
                            #writer.add_scalar("training loss", loss, iterat)
                   
                            iterat += 1
                            
                    count += 1
          
                if i % 10 == 0:
                    print ("Epsilon is = ", agent.epsilon)
                    print (i)
                
        
                if i % 100 == 0:
                    T.save(agent.q_eval.state_dict(), "Model"+str(i)+".pth")
                    
     


def test(K, dim, costum_grid, model_dir):
    
    K= 4
    agent = Agent(dim = dim, K = K, gamma = 1, epsilon = 1, \
                  lr= 0.001, mem_size = 5000, batch_size = 32,\
                      eps_min = 0.01 , eps_dec = 1.25/5000, replace=10)

    agent.q_eval.load_state_dict(T.load(model_dir))

        

    agent.epsilon = 0
    
    Q_list = []
    Ahat_list = []
    A_list = []
    done = False
    
    
    if costum_grid!=None:
        grid_ = copy.deepcopy(costum_grid)
        grid_gr  = copy.deepcopy(grid_)
        
    else:
        
        grid_ = rand_grid_gen(None)
        grid_gr  = copy.deepcopy(grid_)
        
    while not done:
        
        observation = grid_.data
        #action = agent.choose_action(observation, grid_.viol_nodes()[0])
        with T.no_grad():
                
            Q, advantage = agent.q_eval.forward(observation)

            A_list.append(advantage)
            Q_list.append(Q)
            Ahat_list.append(advantage-advantage.max())
            viol_nodes = grid_.viol_nodes()[0]
            action = viol_nodes[T.argmax(Q[viol_nodes]).item()]
            
        # print ("VIOLS", grid_.viol_nodes()[0])
        # print (agent.q_eval.forward(grid_.data))
        grid_.coarsen_node(action)
        done = True if grid_.viol_nodes()[2] == 0 else False
        
    print ("RL result", sum(grid_.active)/grid_.num_nodes)
    #grid_.plot()
    
    grid_gr = greedy_coarsening(grid_gr)
    
    return  grid_, grid_gr, Q_list, A_list, Ahat_list

#gr, rl = test(0.1)

def node_hop_neigh(K, node, list_neighbours):
    
    set_all = set([])
    
    set_all  = set_all.union(set([node]))
    prev_set = copy.deepcopy(set_all)
    this_hop = [node]

    for i in range(K):
        
        for node in this_hop:
            
            set_all = set_all.union(set(list_neighbours[node]))
            
        this_hop = list(set_all.difference(prev_set))
        prev_set = copy.deepcopy(set_all)
    
    return list(set_all)
 
def regional_update_test (given_grid, K, model_dir, Test_greedy = True):
    
    if given_grid == None:
        grid_ = rand_grid_gen(None)
        
    else:
        grid_ = copy.deepcopy(given_grid)
        
    if Test_greedy:
        
        grid_gr  = copy.deepcopy(grid_)

    
    agent = Agent(dim = 32, K = K, gamma = 1, epsilon = 1, \
                      lr= 0.001, mem_size = 5000, batch_size = 64, \
                          eps_min = 0.01 , eps_dec = 1.333/5000, replace=10)
    
    
    agent.q_eval.load_state_dict(T.load(model_dir))
        
    
    agent.epsilon = 0
    
    done = False
    
    T_start = time.time()
        
    observation = grid_.data
    with T.no_grad():
        Q, advantage = agent.q_eval.forward(observation)
        
    
    adv_tensor = copy.deepcopy(advantage)  #get all of the advantage values
    
    list_neighbours = grid_.list_neighbours
    ##get k-hop neighbours of every violating node
    all_viols = grid_.violating_nodes
    while not done:

        node_max = all_viols [T.argmax(adv_tensor[all_viols ])]

        newly_removed = grid_.coarsen_node(node_max)
        all_viols = list(set(all_viols)-set(newly_removed))
        #print (len(list_neighbours),list_neighbours, node_max)
        k_hop  = node_hop_neigh(K, node_max, list_neighbours[0])
        k2_hop = node_hop_neigh(2*K, node_max, list_neighbours[0])

        observation = grid_.subgrid(k2_hop)

        #action = agent.choose_action(observation, grid_.viol_nodes()[0])
        with T.no_grad():
                
            _, advantage = agent.q_eval.forward(observation)
            
        update_list = [k2_hop.index(aa) for aa in k_hop]

        adv_tensor[k_hop] = advantage[update_list]

        done = True if len(all_viols) == 0 else False
      
    T_end  = time.time()
    computation_time = T_end-T_start
    
    rl_ffrac = sum(grid_.active)/grid_.num_nodes 
    print ("RL result", sum(grid_.active)/grid_.num_nodes)
    print ("number of nodes = ", grid_.num_nodes)
        
    if Test_greedy:   
        grid_gr = greedy_coarsening(grid_gr) 
        gr_ffrac = sum(grid_gr.active)/grid_gr.num_nodes
    
    if Test_greedy:
        return  grid_, rl_ffrac, grid_gr, gr_ffrac, computation_time
    else:
        return grid_, rl_ffrac, computation_time
