import numpy as np
import scipy
from scipy import sparse
import copy
from Unstructured import grid

def cycle(grid_):
    
    init_fpts = grid_.fine_nodes
    init_cpts = grid_.coarse_nodes
    A = grid_.A#.toarray()
    min_dominance = 0.56
    eps = (2.0 - 2.0*min_dominance) / (2.0*min_dominance - 1.0)
    sigma = 2.0 / (2.0 + eps)
    #h = 1.0/(no_nodes_1d + 1)
    #A = (1.0/h**2) * A
    ## form B, separating the F- and C-points.
    B = sparse.csr_matrix(A.shape)
    #lenB = sa.N
    lenB = B.shape[0]
    #sqrt_lenB = sa.sqrt_N
    #sqrt_lenB = int(np.sqrt(lenB))
    lenfpts = len(init_fpts)
    lencpts = len(init_cpts)
    fcpts_perm = np.zeros(A.shape[0], dtype='int64')
    fcpts_perm[0:lenfpts] = init_fpts
    fcpts_perm[lenfpts:] = init_cpts
    Bff = A[np.ix_(init_fpts, init_fpts)]
    B = A[fcpts_perm[:, None], fcpts_perm]
    ## Calculate theta-i and Dff
    theta_inv = np.sum(np.abs(Bff), axis=0) / Bff.diagonal()
    Dff = np.multiply(2.0-theta_inv, Bff.diagonal())  # (2.0-theta_inv)*Bff.diagonal()
    Dffinv = 1.0 / Dff
    #Dffinv = np.array(1.0 / Dff)[0]
    ## Minv from reduction based AMG (REL)
    Minv = sparse.csr_matrix(A.shape)
    Minv[np.arange(lenfpts), np.arange(lenfpts)]= sigma*Dffinv
    ## Form P, the interpolation matrix
    DinvBfc = -B[0:lenfpts, lenfpts:lenB]
    DinvBfc = sparse.spdiags(Dffinv,0,lenfpts,lenfpts, format='csr').dot(DinvBfc)
    P = sparse.csr_matrix((lenB,lencpts))
    P[0:lenfpts, :] = DinvBfc
    P[lenfpts:lenB, :] = sparse.identity(lencpts, format='csr')
    #B = scipy.sparse.csr_matrix(B)
    AA = (P.transpose()).dot(B.dot(P))
    
    return AA

    
def make_coarse_grid(grid_):
    
    c_nodes = grid_.coarse_nodes
    points = np.concatenate((np.expand_dims(grid_.mesh.X, axis=1), np.expand_dims(grid_.mesh.Y, axis=1)), axis=1)
    c_V = points[c_nodes]
    c_nv = c_V.shape[0]
    c_X = grid_.mesh.X[c_nodes]
    c_Y = grid_.mesh.Y[c_nodes]
    
    old_E = np.array(grid_.mesh.E)
    list_c_E = []
    for i in range(len(grid_.mesh.E)):
        if old_E[i,0] in c_nodes and old_E[i,1] in c_nodes and old_E[i,2] in c_nodes:
            list_c_E.append(i)
           
    c_E = old_E[list_c_E]
    c_ne = c_E.shape[0]
    
    c_mesh = copy.deepcopy(grid_.mesh)
    c_mesh.X = c_X
    c_mesh.Y = c_Y
    c_mesh.V = c_V
    c_mesh.E = c_E
    c_mesh.nv = c_nv
    c_mesh.ne = c_ne
    c_A = cycle(grid_)
    
    c_rows = c_A.tocoo().row
    c_cols = c_A.tocoo().col
    
    c_Edges = np.concatenate((np.expand_dims(c_rows, axis=1), np.expand_dims(c_cols, axis=1)), axis=1)
    c_Edges = list(tuple(map(tuple, c_Edges)))
    
    
    c_N = [i for i in range(c_X.shape[0])]
    c_num_edges = len(c_Edges)
    
    c_mesh.Edges = c_Edges
    c_mesh.N = c_N
    c_mesh.num_edges = c_num_edges
        
    
    
    c_fine_nodes = [i for i in range(c_nv)]
    c_coarse_nodes = []
    Theta = grid_.Theta
    
    coarse_grid = grid(c_A, c_fine_nodes, c_coarse_nodes, c_mesh, Theta)
    
    return coarse_grid
