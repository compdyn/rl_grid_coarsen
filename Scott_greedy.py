import numpy as np
from pyamg.gallery import stencil_grid
import fem
import random
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pygmsh
from  Unstructured import MyMesh, grid



def rand_Amesh_gen(mesh):
    
    num_Qhull_nodes = random.randint(20, 40)
    points = np.random.rand(num_Qhull_nodes, 2)            # 30 random points in 2-D
    hull = ConvexHull(points)
    
    msh_sz = 0.1*random.random()+0.2
    
    if mesh == None:
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(
                
                    hull.points[hull.vertices.tolist()].tolist()
                ,
                mesh_size=msh_sz,
            )
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


def diag_dominance(A):
    """Returns the measures of the diagonal dominance for each row of a matrix"""
    no_rows = A.shape[0]
    dominance = np.zeros( (no_rows,) )
    for i in range(no_rows):
        dominance[i] = np.absolute(A[i,i]) / np.sum(np.absolute(A[i,:]))
    return dominance


def greedy_alg(A, theta):
    """Returns the set of fine and coarse grid points and number of fine grid points"""
    A = A.todense()
    dominance = diag_dominance(A)
    length = len(dominance)
    F = np.array([], dtype=int)
    U = np.arange(length, dtype=int)
    C = np.array([], dtype=int)
    for i in range(length):
        if dominance[i] >= theta:
            F = np.append(F, i)
    U = np.setdiff1d(U, F)
    while len(U) >= 1:
        c =np.argmin(dominance[U])
        C = np.append(C,U[c])
        l = np.nonzero(A[U[c],:])[1]
        U = np.setdiff1d(U, U[c])
        for i in np.intersect1d(U, l):
            sum1 = np.sum(np.absolute(A[i,U]))
            sum2 = np.sum(np.absolute(A[i,F]))
            dominance[i] = np.absolute(A[i,i]) / (sum1 + sum2)
            if dominance[i] >= theta:
                F = np.append(F, i)
                U = np.setdiff1d(U, F)
    return len(F), F, C
#A = poisson( (64,64) )
A = stencil_grid([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], (32, 32), dtype=float, format='csr')
[l, F, C] = greedy_alg(A, 0.56)
#print('No. of nodes in fine grid:',l)
#print('F :',F)
#print('C :',C)

def greedy_coarsening(grid_):
    
    if grid_ == None:
        grid_ = rand_grid_gen(None)
    
    A = grid_.A
    num_f, fine, coarse = greedy_alg(A, 0.56)
    for node in coarse:
        grid_.coarsen_node(node)
    
    grid_.fine_nodes = grid_.active.nonzero()[0].tolist()#list(set(grid_.fine_nodes)-set(maxes))
    grid_.violating_nodes = grid_.is_violating.nonzero()[0].tolist()
    #print ("Greedy Algorithm Result is", sum(grid_.active)/grid_.num_nodes)  
    #grid.plot()
    return grid_
    