import numpy as np
import torch as T
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x, global_mean_pool,
                                BatchNorm, InstanceNorm)
import random
import scipy as sp
from pyamg.gallery.diffusion import diffusion_stencil_2d
from pyamg.gallery import stencil_grid
from torch_geometric.data import Data
from pyamg.aggregation import lloyd_aggregation
from pyamg.gallery import poisson
from scipy.sparse import coo_matrix
import time 
from torch_geometric.data import Data, DataLoader
from MG_Agent import Agent
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pygmsh
from  Unstructured import MyMesh, grid, rand_Amesh_gen, rand_grid_gen, plot_cycle
import fem 
from torch.utils.tensorboard import SummaryWriter
import sys
from Scott_greedy import greedy_coarsening
from scipy.spatial import Delaunay
import copy 
from Cycle import make_coarse_grid

def Coloring(graph, regions, list_neighbours):
    
    all_colors = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    colors = [[] for i in range(graph.x.shape[0])]
    reg_color = [[] for i in range(len(regions))]
    for i in range(len(regions)):
        
        reg = regions[i]
        forbid = []
        
        
        for node in reg:
            for neigh in list_neighbours[node]:
                if len(colors[neigh]) != 0:
                    forbid.append(colors[neigh][0])
                    forbid = list(set(forbid))
                    #print (reg_list4nodes[neigh])
        
        #print (forbid)
        available = list(set(all_colors) - set(forbid))
        
        for node in reg:
            
            colors[node].append(min(available))
        
        reg_color[i].append(min(available))
        
    return colors, reg_color

def hop_neigh(K, region, list_neighbours):
    
    set_all = set([])
    
    set_all = set_all.union(set(region))
    prev_set = copy.deepcopy(set_all)
    this_hop = region
    
    for i in range(K):
        
        for node in this_hop:
            
            set_all = set_all.union(set(list_neighbours[node]))
            
        this_hop = list(set_all.difference(prev_set))
        prev_set = copy.deepcopy(set_all)
    
    return list(set_all)

def Lloyd(given_grid, Test_greedy = True):
    if given_grid == None:
        grid_ = rand_grid_gen(None)
    else:
        grid_ = copy.deepcopy(given_grid)
    grid_gr = copy.deepcopy(grid_)
    AA = grid_.A
    AA = sp.sparse.csr_matrix(AA)
    num_nodes = grid_.num_nodes
    num_C = int(num_nodes/30)
    Ratio = 0.05#num_C/(num_nodes)
    K = 7
    
    Agg = lloyd_aggregation(AA,ratio=Ratio,maxiter=1000)[0]
    
    AA = sp.sparse.csr_matrix.toarray(Agg)
    AA = T.from_numpy(abs(AA))
    num_C = AA.shape[1]
    
    
    regions = []
    hop_regs = []
    for i in range(num_C):
        
        regions.append(T.nonzero(AA[:,i]).flatten().tolist())
        
    
        
    list_neighbours = [[] for i in range(grid_.x.shape[0])]
    
    for i in range(grid_.edge_index.shape[1]):
    
        list_neighbours[grid_.edge_index[0,i].clone().tolist()].append(grid_.edge_index[1\
                                                                      ,i].clone().tolist()) 
        
            
    for i in range(len(regions)):
        
        hop_regs.append(hop_neigh(K, regions[i], list_neighbours))
        
        
    #colors, reg_color = Coloring(grid_, regions, list_neighbours)
    mymsh = grid_.mesh
    points = mymsh.V
    #tri = Delaunay(points)
    #plt.triplot(points[:,0], points[:,1], tri.simplices)
    
    color_code = {0.0:'b.', 1.0:'g.', 2.0:'r.', 3.0:'k.', 4.0: 'y.', 5.0:'c.', 6.0:'m.'}
    
    #for i in range(len(regions)):
        
        #plt.plot(points[regions[i],0], points[regions[i],1], color_code.get(reg_color[i][0]))
        
        
    agent = Agent(dim = 32, K = 12, gamma = 1, epsilon = 1, \
                      lr= 0.001, mem_size = 5000, batch_size = 64, net_type = 'TAGConv',  \
                          eps_min = 0.01 , eps_dec = 1.333/5000, replace=10)
        
        
        
    agent.q_eval.load_state_dict(T.load('Models/Dueling_batch_train_final.pth'))
    agent.epsilon = 0
    '''
    num_Qhull_nodes = random.randint(15, 45)
    points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
    hull = ConvexHull(points)
    
    msh_sz = 0.1*random.random()+0.2
    
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(
            
                hull.points[hull.vertices.tolist()].tolist()
            ,
            mesh_size=msh_sz,
        )
        mesh = geom.generate_mesh()
    '''
    #mesh = T.load("mesh.pth")
    done = False
    '''
    grid_ = rand_grid_gen(mesh)
    grid_gr  = rand_grid_gen(mesh)
    '''
    scores = np.zeros(len(regions))
    
    for i in range(len(regions)):
        
        scores[i] = len(regions[i])
    t1 = time.time()
   
    while not done:
        
        max_idx = np.argmax(scores)
        hreg = hop_regs[max_idx]
        reg = regions[max_idx]
        observation = grid_.subgrid(hreg)
        is_viol = grid_.viol_nodes()[1]
        hops = list(set(hreg).difference(set(reg)))
        is_viol[hops] = 0
        sub_viols = T.nonzero(is_viol[hreg]).flatten().tolist()
        if sub_viols == []:
            scores[max_idx] = -1
        
        else:
            act = agent.choose_action(observation, sub_viols)
            action = hreg[act]
            # print ("ACTION", action)
            # print ("VIOLS", grid_.viol_nodes()[0])
            # print (agent.q_eval.forward(grid_.data))
            grid_.coarsen_node(action)
            
            new_is_viol = grid_.viol_nodes()[1]
            scores[max_idx] = T.sum(new_is_viol[reg]).item()
            done = True if grid_.viol_nodes()[2] == 0 else False
        
    print ("RL result", sum(grid_.active)/grid_.num_nodes)
    #grid_.plot()
    t2 = time.time()
    print ("#nodes",grid_.num_nodes, "TIME", t2-t1)
    
    
    if Test_greedy:
        
        grid_gr = greedy_coarsening(grid_gr)
    
    return grid_, grid_gr
    


def Multilevel_MG (given_grid, num_cycle, Plot=False, Test_greedy=False):
    
    rl_list = []
    rl_f_frac = []
    gr_list = []
    gr_f_frac = []
    
    if given_grid == None:
        given_grid = rand_grid_gen(None)
    
    crl = copy.deepcopy(given_grid)
    
    for i in range(num_cycle):
        rl,_ = Lloyd(crl)
        
        rl_list.append(copy.deepcopy(rl))
        rl_f_frac.append(sum(rl_list[i].active)/rl_list[i].num_nodes)
        crl = make_coarse_grid(rl)
        
    crl = copy.deepcopy(given_grid)
    
    if Test_greedy:
        
        for i in range(num_cycle):
            
            _,gr = Lloyd(crl)
            gr_list.append(copy.deepcopy(gr))
            gr_f_frac.append(sum(gr_list[i].active)/gr_list[i].num_nodes)
            crl = make_coarse_grid(gr)
        
    
    if Plot:
        
        plot_cycle(rl_list)
        title = 'RL '+ str(num_cycle-1)+' level coarsening, black edges = before coarsening,\n green edges = after cycle 1, (if 2 level coarsening, magenta edges = \n after cycle 2). F-fractions of cycles = '
        for i in range(num_cycle):
            title += str(np.round(rl_f_frac[i],4)) + ', '
        
        plt.title(title)
        plt.show()
        
        if Test_greedy:
            plot_cycle(gr_list)
            title = 'Greedy '+ str(num_cycle-1)+' level coarsening, black edges = before coarsening,\n green edges = after cycle 1, (if 2 level coarsening, magenta edges = \n after cycle 2). F-fractions of cycles = '
            for i in range(num_cycle):
                title += str(np.round(gr_f_frac[i],4)) + ', '
            
            plt.title(title)
            plt.show()
            
    return rl_list, gr_list
        
        
    
#rl_list, gr_list = Multilevel_MG (None, 2, Plot=True,Test_greedy=True)
