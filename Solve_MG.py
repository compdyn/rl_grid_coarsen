import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation, PillowWriter  
import random
import torch
import scipy
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla
from Lloyd_Unstructured import Lloyd
import copy
from  Unstructured import MyMesh, grid, rand_Amesh_gen, rand_grid_gen, \
    plot_cycle, structured

from Cycle import make_coarse_grid
import matplotlib.pyplot as plt
import sys
from MG_Agent import Agent
from Optim import Linear_Coarsening_Lloyd
import time 
from Unstructured import MyMesh


class To_MyMesh:
    def __init__(self, tri):
        
        self.X = tri.points[:,0:1].flatten()
        self.Y = tri.points[:,1:2].flatten()
        self.E = tri.simplices
        self.V = tri.points[:,0:2]
        self.nv = tri.points[:,0:2].shape[0]
        self.ne = len(tri.simplices)
        
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
        
        
## Relaxation
def relax_wjacobif(B,f,u,Minv):
    Dii = Minv.diagonal()
    r = f - B.dot(u)
    u = u + r*Dii
    return u

def is_pos_def(x):
    
    out = np.linalg.eigvals(x)
    print (out.min())
    return  out

##### V cycle without recursion
#### -uxx-uyy = 0
def nr_vcycle_amgr(tot_num_loop, n_cut, min_dominance, level, no_prerelax, no_postrelax, grid_, precalc,\
                   name_, Greedy=False):
    ## Calculate epsilon and sigma
    #min_dominance = 0.56
    grid_list = []
    eps = (2.0 - 2.0*min_dominance) / (2.0*min_dominance - 1.0)
    sigma = 2.0 / (2.0 + eps)
    #level = 2
    lvlm1 = level - 1
    A = level * [None]
    fpts = lvlm1 * [None]
    cpts = lvlm1 * [None]
    fcpts_perm = lvlm1 * [None]
    B = lvlm1 * [None]
    lenB = lvlm1 * [None]
    lenfpts = lvlm1 * [None]
    lencpts = lvlm1 * [None]
    #sqrt_lenB = lvlm1 * [None]
    Bff = lvlm1 * [None]
    theta_inv = lvlm1 * [None]
    Dff = lvlm1 * [None]
    Dffinv = lvlm1 * [None]
    Minv = lvlm1 * [None]
    DinvBfc = lvlm1 * [None]
    P = lvlm1 * [None]    
    v = level * [None]
    f = level * [None]
    x = lvlm1 * [None]
    b = lvlm1 * [None]
    count_nonzeros_A = level * [None]
    count_grid_points = level * [None]
    
    #A0 = np.load('A0.npy', allow_pickle=True).tolist()
    A0 = grid_.A
    count_nonzeros_A[0] = A0.count_nonzero()
    count_grid_points[0] = A0.shape[0]
    #no_prerelax = 1
    #no_postrelax = 1
    two_norm_old = 1.0
    ### Finest Grid
    ## form A, and run CGsel to have the F-pts and C-pts
    A[0] = A0

    #A[0] = np.load('A0.npy', allow_pickle=True).tolist()
    ## form all of the required matrices of different levels
    
    agent = Agent(dim = 32, K = 12, gamma = 1, epsilon = 1, \
                      lr= 0.001, mem_size = 5000, net_type = 'TAGConv', batch_size = 64,  \
                          eps_min = 0.01 , eps_dec = 1.333/5000, replace=10)

    agent.q_eval.load_state_dict(torch.load('Models/Dueling_batch_train_final.pth'))

    for i in range(lvlm1):
        
        if Greedy:
            if precalc== None:
                
                grid_ = Linear_Coarsening_Lloyd(grid_, agent, True)
            else:
                
                grid_ = torch.load(name_)
                        
        else:
            
            if precalc == None:
                
                grid_ = Linear_Coarsening_Lloyd(grid_, agent, False)
                
            else:
                
                grid_ = torch.load(name_)
            
            
        grid_list.append(grid_)
        
        # grid_.plot()
        # plt.show()
            
        # fpts[i] = np.load('Fpts'+str(i)+'.npy', allow_pickle=True).tolist()
        # cpts[i] = np.load('Cpts'+str(i)+'.npy', allow_pickle=True).tolist()
            
        fpts[i] = grid_.fine_nodes
        cpts[i] = grid_.coarse_nodes
        
        ## form B, separating the F- and C-points.
        B[i] = sparse.csr_matrix(A[i].shape)
        #lenB[i] = sa.N
        lenB[i] = B[i].shape[0]
        #sqrt_lenB[i] = sa.sqrt_N
        #sqrt_lenB[i] = int(np.sqrt(lenB[i]))
        lenfpts[i] = len(fpts[i])
        lencpts[i] = len(cpts[i])
        
        Fl = [j for j in range(lenfpts[i])]
        Cl = [j+lenfpts[i] for j in range(lencpts[i])]
        
    
        fcpts_perm[i] = np.zeros(A[i].shape[0], dtype='int64')

        fcpts_perm[i][0:lenfpts[i]] = fpts[i]
        fcpts_perm[i][lenfpts[i]:] = cpts[i]
        
        Bff[i] = A[i][np.ix_(fpts[i], fpts[i])]
        
        #Ali Commented#
        #B[i] = A[i][fcpts_perm[i][:, None], fcpts_perm[i]]
        ###############
       
        # B[i][:,0:lenfpts[i]][0:lenfpts[i],:] = A[i][:,fpts[i]][fpts[i],:]
        # B[i][:,0:lenfpts[i]][ lenfpts[i]:,:] = A[i][:,fpts[i]][cpts[i],:]
        # B[i][:,lenfpts[i]:][ 0:lenfpts[i],:] = A[i][:,cpts[i]][fpts[i],:]
        # B[i][:,lenfpts[i]:][ lenfpts[i]:,:] = A[i][:,cpts[i]][cpts[i],:]
        
        
        Bff[i] = A[i][np.ix_(fpts[i], fpts[i])]
        B[i][np.ix_(Fl, Fl)] = Bff[i]
        B[i][np.ix_(Cl, Fl)] = A[i][np.ix_(cpts[i], fpts[i])]
        B[i][np.ix_(Fl, Cl)] = A[i][np.ix_(fpts[i], cpts[i])]
        B[i][np.ix_(Cl, Cl)] = A[i][np.ix_(cpts[i], cpts[i])]
        ## Calculate theta-i and Dff
        theta_inv[i] = np.sum(np.abs(Bff[i]), axis=0) / Bff[i].diagonal()
        Dff[i] = np.multiply(2.0-theta_inv[i], Bff[i].diagonal())  # (2.0-theta_inv)*Bff.diagonal()
        print (min(np.array(Dff[0]).flatten()))
        Dffinv[i] = 1.0 / Dff[i]
        
        #Dffinv[i] = np.array(1.0 / Dff[i])[0]

        ## Minv from reduction based AMG (REL)
        Minv[i] = sparse.csr_matrix(A[i].shape)
        Minv[i][np.ix_(Fl,Fl)]= sigma*Dffinv[i]

        ## Form P, the interpolation matrix
        
        ##Ali Commented##
        # DinvBfc[i] = -B[i][0:lenfpts[i], lenfpts[i]:]
        
        #################
        
        ##Ali Added##
        DinvBfc[i] = -B[i][np.ix_(Fl, Cl)]
        
        
        DinvBfc[i] = sparse.spdiags(Dffinv[i],0,lenfpts[i],lenfpts[i], format='csr').dot(DinvBfc[i])
        
        P[i] = sparse.csr_matrix((lenB[i],lencpts[i]))
        P[i][0:lenfpts[i], :] = DinvBfc[i]
        P[i][lenfpts[i]:lenB[i], :] = sparse.identity(lencpts[i], format='csr')
        ## form v, x, and b
        v[i] = np.zeros(lenB[i])
        if i > 0:
            x[i] = np.zeros(lenB[i])
        b[i] = np.zeros(lenB[i])

        ## form A2h
        A[i+1] = (P[i].transpose()).dot(B[i].dot(P[i]))
        
        #count_nonzeros_A[i+1] = A[i+1].count_nonzero()
        #count_grid_points[i+1] = A[i+1].shape[0]

    
  
            
        if abs(np.linalg.eigvals(A[-1].toarray()).min())<1e-6:
            
            A[-1][-1] = sparse.csr_matrix(A[-1][-1].toarray()*0+1)
                
    # for i in range(len(B)):
            
    #         if abs(np.linalg.eigvals(B[i].toarray()).min())<1e-6:
                
    #             B[i][-1] = sparse.csr_matrix(B[i][-1].toarray()*0+1) 
    #             print("KKHHHHAAAIII", abs(np.linalg.eigvals(B[i].toarray()).min()))
                
              
    ## Exact solution
    xexact = np.zeros(lenB[0])

    ## x, f, b for the finest grid
    random_state = np.random.RandomState(seed=5)
    x[0] = random_state.normal(size=lenB[0])
    
    #x[0] = np.random.normal(size=lenB[0])
    #x[0] = np.zeros(lenB[0])
    f[0] = np.zeros(lenB[0])
    b[0][0:lenfpts[0]] = f[0][fpts[0]]
    b[0][lenfpts[0]:lenB[0]] = f[0][cpts[0]]


    list_v = []
    list_x = []
    mean_v = []
    #1000 for 5617, 22241; 800 for 1433; for dom2: all 1000
    
    A_inv = sla.inv(A[-1])


    two_norm_n_list = []
    convfacts = [0]
    for no_loop in range(tot_num_loop+1):
    ## go from finest to coarsest
        
            
            
        list_x.append(x[0])
        for i in range(lvlm1):
            for j in range(no_prerelax):
                
                x[i] = relax_wjacobif(B[i],b[i],x[i],Minv[i])
            
                
            f[i+1] = (P[i].transpose()).dot(b[i] - B[i].dot(x[i]))
            
                
            if i <= lvlm1-2:
                
                b[i+1][0:lenfpts[i+1]] = f[i+1][fpts[i+1]]
                b[i+1][lenfpts[i+1]:lenB[i+1]] = f[i+1][cpts[i+1]]
                
        ## solve for the coarsest
        v[-1] = (A_inv).dot(f[-1])
    
        ## go from coarsest to finest
        for i in range(lvlm1-1, -1, -1):
            
            x[i] = x[i] + P[i].dot(v[i+1])

            
            for j in range(no_postrelax):
                
                x[i] = relax_wjacobif(B[i],b[i],x[i],Minv[i])
      
            v[i][fpts[i]] = x[i][0:lenfpts[i]]
            v[i][cpts[i]] = x[i][lenfpts[i]:lenB[i]]
     
        
        for v_indx in range(len(v)):
            
            v[v_indx] = v[v_indx] - v[v_indx].mean()
            
        #inf_norm = np.linalg.norm(xexact - v, np.inf)
        two_norm_n = np.linalg.norm(xexact - v[0]+v[0].mean(), 2)
        ratio_norm = two_norm_n / two_norm_old
        two_norm_old = two_norm_n
        #print('ratio of norms:', ratio_norm)
        list_v.append(copy.deepcopy(v[0]-v[0].mean()))
        mean_v.append((v[0]-v[0].mean()).mean())
        
        two_norm_n_list.append(two_norm_n)
        
        if len(two_norm_n_list)>=15:
            
            factor = (two_norm_n_list[-1]/two_norm_n_list[-10])**(1/10)
            if abs(factor - convfacts[-1])<0.01 and abs(factor - convfacts[-2])<0.01\
                and abs(factor - convfacts[-3])<0.01:
                    
                Grid_Complexity = 0
                Operator_Complexity = 0
                for i in range(len(A)):
                    
                    Operator_Complexity += np.nonzero(A[i])[0].shape[0]
                    Grid_Complexity += A[i].shape[0]
                    
                Grid_Complexity = Grid_Complexity/A[0].shape[0]
                Operator_Complexity = Operator_Complexity/np.nonzero(A[0])[0].shape[0]
                print ("factor = ", factor, "num iter", no_loop)
                
                plt.plot(np.arange(0,v[0].shape[0]), v[0])
                plt.title('solution on the fine grid after convergence factor stops changing, solving for Ax = 0')
                plt.ylim([-2, 2])
                plt.xlabel('nodes')
                plt.ylabel('nodes values')
                plt.show()
                
                plt.plot(np.arange(0,v[1].shape[0]), v[1])
                plt.title('solution on the coarse grid after convergence factor stops changing, solving for Ax = 0')
                plt.ylim([-2, 2])
                plt.xlabel('nodes')
                plt.ylabel('nodes values')
                plt.show()
                
                
                # kir
                return factor, convfacts, Grid_Complexity, Operator_Complexity
            
            if no_loop == tot_num_loop:
                
                Grid_Complexity = 0
                Operator_Complexity = 0
                for i in range(len(A)):
                    
                    Operator_Complexity += np.nonzero(A[i])[0].shape[0]
                    Grid_Complexity += A[i].shape[0]
                    
                Grid_Complexity = Grid_Complexity/A[0].shape[0]
                Operator_Complexity = Operator_Complexity/np.nonzero(A[0])[0].shape[0]
                print ("factor = ", factor, "num iter", no_loop)
                
                plt.plot(np.arange(0,v[0].shape[0]), v[0])
                plt.title('solution on the fine grid after convergence factor stops changing, solving for Ax = 0')
                plt.ylim([-2, 2])
                plt.xlabel('nodes')
                plt.ylabel('nodes values')
                plt.show()
                
                plt.plot(np.arange(0,v[1].shape[0]), v[1])
                plt.title('solution on the coarse grid after convergence factor stops changing, solving for Ax = 0')
                plt.ylim([-2, 2])
                plt.xlabel('nodes')
                plt.ylabel('nodes values')
                plt.show()
                
                
                # kir
                return factor, convfacts, Grid_Complexity, Operator_Complexity
                
            convfacts.append(factor)
            
        if no_loop == 0:
            firstv = copy.deepcopy(v[0]-v[0].mean())
            
       
        for ij in range(len(v)):
            
            v[ij] = v[ij] - v[ij].mean()
        
    #print('convergence factor using consecutive norms:', ratio_norm)
    
    conv_fac_ini_fin = (two_norm_n_list[-1]/two_norm_n_list[-10])**(1/10)
    lastv = v[0]-v[0].mean()
    
    # plt.plot(np.arange(0, 1, 1/xexact.shape[0]), lastv)
    # plt.ylim([-2, 2])
    # plt.show()
    
    # plt.plot(np.arange(0,len(mean_v)), mean_v)
    # plt.title(Greedy)
    # plt.ylim([-2, 2])
    # plt.show()
    
    print('convergence factor using init and final:', conv_fac_ini_fin)
    Grid_Complexity = 0
    Operator_Complexity = 0
    for i in range(len(A)):
        
        Operator_Complexity += np.nonzero(A[i])[0].shape[0]
        Grid_Complexity += A[i].shape[0]
        
    Grid_Complexity = Grid_Complexity/A[0].shape[0]
    Operator_Complexity = Operator_Complexity/np.nonzero(A[0])[0].shape[0]
        
    # return firstv, lastv, list_v, list_x 
    return conv_fac_ini_fin, convfacts, Grid_Complexity, Operator_Complexity
    

list_RL_CRate  = []
list_GR_CRate  = []
list_GR_Gcompx = []
list_RL_Gcompx = []
list_GR_Gcompx = []
list_RL_Ocompx = []
list_GR_Ocompx = []


dict_rate_complexity = torch.load('rate_and_complx.pth')

# dict_rate_complexity["Convex"] = {}




#dict_rate_complexity["Graded_mesh"] = {}
# l = [2,8,10,11]
# for i in l:
#     print ("convex, number ",i)
#     precalc = i

    
#     grid_ = torch.load('Test_Graphs/Hand_crafted/Grids/Graph_'+str(precalc))
#     grid_.A = -grid_.A
#     torch.save(grid_, 'Test_Graphs/Hand_crafted/Grids/Graph_'+str(precalc))
    
#     rl = torch.load('Test_Graphs/Hand_crafted/Grids/rl_1L_'+str(precalc))
#     rl.A = -rl.A
#     torch.save(rl, 'Test_Graphs/Hand_crafted/Grids/rl_1L_'+str(precalc))
    
#     gr = torch.load('Test_Graphs/Hand_crafted/Grids/gr_1L_'+str(precalc))
#     gr.A = -gr.A
#     torch.save(gr, 'Test_Graphs/Hand_crafted/Grids/gr_1L_'+str(precalc))
    
'''
for i in range(1,13):
    print ("Graded_mesh, number ",i)
    precalc = i
    n_loop  = 100
    n_cut   = 10
    
    grid_ = torch.load('Test_Graphs/Smoothly/Graph'+str(precalc))
    grid_gr = copy.deepcopy(grid_)
    
    num_nodes = grid_.num_nodes
    
    rl_cr, rl_cr_list, RL_Gcompx, RL_Ocompx = nr_vcycle_amgr(n_loop, n_cut, 0.56, 2, 2, 2, grid_, \
                                                 precalc, \
                                                     'Test_Graphs/Smoothly/RL_1L_'+str(precalc), Greedy=False)
    
    
    gr_cr, gr_cr_list, GR_Gcompx, GR_Ocompx = nr_vcycle_amgr(n_loop, n_cut,0.56, 2, 2, 2, grid_gr, \
                                                 precalc, \
                                                     'Test_Graphs/Smoothly/GR_1L_'+str(precalc), Greedy=True)
    
    dict_rate_complexity['Graded_mesh'][i] = {'num_nodes':num_nodes, 'rl_cr': rl_cr, 'gr_cr': gr_cr, \
                                               'RL_Gcompx': RL_Gcompx, 'GR_Gcompx':GR_Gcompx,
                                 'RL_Ocompx':RL_Ocompx, 'GR_Ocompx':GR_Ocompx}
    
    torch.save(dict_rate_complexity, 'rate_and_complx.pth')
    
    

dict_rate_complexity["Wide_valence"] = {}
for i in range(1,13):
    
    print ("Wide_valence, number ",i)
    precalc = i
    n_loop  = 100
    n_cut   = 10
    
    grid_ = torch.load('Test_Graphs/test_STD/Graph'+str(precalc))
    grid_gr = copy.deepcopy(grid_)
    
    num_nodes = grid_.num_nodes
    
    rl_cr, rl_cr_list, RL_Gcompx, RL_Ocompx = nr_vcycle_amgr(n_loop, n_cut, 0.56, 2, 2, 2, grid_, \
                                                 precalc, \
                                                     'Test_Graphs/test_STD/RL_1L_'+str(precalc), Greedy=False)
    
    gr_cr, gr_cr_list, GR_Gcompx, GR_Ocompx = nr_vcycle_amgr(n_loop, n_cut,0.56, 2, 2, 2, grid_gr, \
                                                 precalc, \
                                                     'Test_Graphs/test_STD/GR_1L_'+str(precalc), Greedy=True)
    
    dict_rate_complexity['Wide_valence'][i] = {'num_nodes':num_nodes, 'rl_cr': rl_cr, 'gr_cr': gr_cr, \
                                               'RL_Gcompx': RL_Gcompx, 'GR_Gcompx':GR_Gcompx,
                                 'RL_Ocompx':RL_Ocompx, 'GR_Ocompx':GR_Ocompx}
    
    torch.save(dict_rate_complexity, 'rate_and_complx.pth')
    
    '''
#dict_rate_complexity["Structured"] = {}
for i in range(1,11):
    
    print ("Structured, number ",i)
    precalc = i
    n_loop  = 100
    n_cut   = 10
    
    grid_ = torch.load('Test_Graphs/Structured/Graph'+str(precalc))
    grid_gr = copy.deepcopy(grid_)
    
    num_nodes = grid_.num_nodes
    
    rl_cr, rl_cr_list, RL_Gcompx, RL_Ocompx = nr_vcycle_amgr(n_loop, n_cut, 0.56, 2, 2, 2, grid_, \
                                                 precalc, \
                                                     'Test_Graphs/Structured/RL_1L_'+str(precalc), Greedy=False)
    
    gr_cr, gr_cr_list, GR_Gcompx, GR_Ocompx = nr_vcycle_amgr(n_loop, n_cut,0.56, 2, 2, 2, grid_gr, \
                                                 precalc, \
                                                     'Test_Graphs/Structured/GR_1L_'+str(precalc), Greedy=True)
    
    dict_rate_complexity['Structured'][i] = {'num_nodes':num_nodes, 'rl_cr': rl_cr, 'gr_cr': gr_cr, \
                                               'RL_Gcompx': RL_Gcompx, 'GR_Gcompx':GR_Gcompx,
                                 'RL_Ocompx':RL_Ocompx, 'GR_Ocompx':GR_Ocompx}
    
    torch.save(dict_rate_complexity, 'rate_and_complx.pth')
    

'''

dict_rate_complexity["Aspect_Ratio"] = {}


for i in range(1,13):
    
    print ("Aspect_Ratio, number ",i)
    precalc = i
    n_loop  = 100
    n_cut   = 10
    
    grid_ = torch.load('Test_Graphs/Aspect_Ratio/Graph_'+str(precalc))
    grid_gr = copy.deepcopy(grid_)
    
    num_nodes = grid_.num_nodes
    
    rl_cr, rl_cr_list, RL_Gcompx, RL_Ocompx = nr_vcycle_amgr(n_loop, n_cut, 0.56, 2, 2, 2, grid_, \
                                                 precalc, \
                                                     'Test_Graphs/Aspect_Ratio/rl_1L_'+str(precalc), Greedy=False)
    
    gr_cr, gr_cr_list, GR_Gcompx, GR_Ocompx = nr_vcycle_amgr(n_loop, n_cut,0.56, 2, 2, 2, grid_gr, \
                                                 precalc, \
                                                     'Test_Graphs/Aspect_Ratio/gr_1L_'+str(precalc), Greedy=True)
    
    dict_rate_complexity['Aspect_Ratio'][i] = {'num_nodes':num_nodes, 'rl_cr': rl_cr, 'gr_cr': gr_cr, \
                                               'RL_Gcompx': RL_Gcompx, 'GR_Gcompx':GR_Gcompx,
                                 'RL_Ocompx':RL_Ocompx, 'GR_Ocompx':GR_Ocompx}
    
    torch.save(dict_rate_complexity, 'rate_and_complx.pth')
    


dict_rate_complexity["Convex"] = {}


for i in range(1,13):
    i = 11
    print ("Convex, number ",i)
    precalc = i
    n_loop  = 100
    n_cut   = 10
    
    grid_ = torch.load('Test_Graphs/Hand_crafted/Grids/Graph_'+str(precalc))
    grid_gr = copy.deepcopy(grid_)
    
    num_nodes = grid_.num_nodes
    
    rl_cr, rl_cr_list, RL_Gcompx, RL_Ocompx = nr_vcycle_amgr(n_loop, n_cut, 0.56, 2, 2, 2, grid_, \
                                                 precalc, \
                                            'Test_Graphs/Hand_crafted/Grids/rl_1L_'+str(precalc), Greedy=False)
    
        
    gr_cr, gr_cr_list, GR_Gcompx, GR_Ocompx = nr_vcycle_amgr(n_loop, n_cut,0.56, 2, 2, 2, grid_gr, \
                                                 precalc, \
                                            'Test_Graphs/Hand_crafted/Grids/gr_1L_'+str(precalc), Greedy=True)
    
    dict_rate_complexity['Convex'][i] = {'num_nodes':num_nodes, 'rl_cr': rl_cr, 'gr_cr': gr_cr, \
                                               'RL_Gcompx': RL_Gcompx, 'GR_Gcompx':GR_Gcompx,
                                 'RL_Ocompx':RL_Ocompx, 'GR_Ocompx':GR_Ocompx}
    
    torch.save(dict_rate_complexity, 'rate_and_complx.pth')
    

dict_rate_complexity["Different_size"] = {}

for j in range(45):
    i = j+1
    if i!=35 and i!=36:
        
        print ("Different_size, number ",i)
        precalc = i
        n_loop  = 100
        n_cut   = 10
        
        grid_ = torch.load('Test_Graphs/Auto_generated/graph_'+str(precalc))
        
        num_nodes = grid_.num_nodes
        
        grid_gr = copy.deepcopy(grid_)
        
        rl_cr, rl_cr_list, RL_Gcompx, RL_Ocompx = nr_vcycle_amgr(n_loop, n_cut, 0.56, 2, 2, 2, grid_, \
                                                 precalc, \
                                                     'Test_Graphs/Auto_generated/rl_1L_'+str(precalc), Greedy=False)
    
        gr_cr, gr_cr_list, GR_Gcompx, GR_Ocompx = nr_vcycle_amgr(n_loop, n_cut,0.56, 2, 2, 2, grid_gr, \
                                                 precalc, \
                                                     'Test_Graphs/Auto_generated/gr_1L_'+str(precalc), Greedy=True)
        
        dict_rate_complexity['Different_size'][i] = {'num_nodes':num_nodes, 'rl_cr': rl_cr, 'gr_cr': gr_cr, \
                                   'RL_Gcompx': RL_Gcompx, 'GR_Gcompx':GR_Gcompx,
                                     'RL_Ocompx':RL_Ocompx, 'GR_Ocompx':GR_Ocompx}
        
        torch.save(dict_rate_complexity, 'rate_and_complx.pth')
    
'''  

