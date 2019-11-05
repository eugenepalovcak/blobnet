""" Growing uniform grid """

import numpy as np
import matplotlib.pyplot as plt
from math import floor
from itertools import product
from scipy.spatial.distance import cdist

def random_point(d=3, loc=0, scale=20):
    return np.random.normal(loc=loc, scale=scale, size=d)

class GrowingUniformGrid(object):
    """ Growing uniform grid object for 3D points 
        points: an Nx3 array of points, zero centered 
  
    Relevant structures in the class include:
    g: the grid, a multidimensional list of cells. Each cell stores 
        a list of integers, which correspond to indeces of points
    index: the list of points idenitified by positional index. 
           A point can be a list, tuple, object, etc. 

    """
    def __init__(self, points, grid_len=50, cells_per_side=5, 
                 occupancy_tol=50, growth_factor=1.5):

        print(type(grid_len)) 
        self.l = grid_len / cells_per_side
        self.o = - grid_len/2
        self.n = cells_per_side - 1
        self.occupancy_tol = occupancy_tol
        self.growth_factor = growth_factor

        self.g = [[[[] for z_cd in range(cells_per_side)]
                       for y_cd in range(cells_per_side)]
                       for x_cd in range(cells_per_side)]

        self.index = []
        self.len_index = 0
       
        self.n_nodes = 0
        self.fill_g(points)

    @property
    def max_occupancy(self):
        """ Finds maximum nodes per cell """
        max_dens = 0
        for i in range(self.n):
           for j in range(self.n):
               for k in range(self.n):
                   max_dens = max(max_dens, len(self.g[i][j][k]))
        return max_dens

    def try_to_rebuild(self):
        """ If a cell has exceeded the maximum allowed occupancy, 
            'grow' the uniform grid so it has finer dimensions. """
        if self.max_occupancy > self.occupancy_tol:
            new_cps = int((self.n+1) * self.growth_factor)
            self.__init__(points=self.index, cells_per_side = new_cps) 
 
        else:
            pass    

    def find_cell(self, cd):
        """ Find the cell for a given point """
        o = self.o
        l = self.l
        n = self.n
        return [min(max(floor((c-o)/l),0), n) for c in cd]

    def add_point(self, cd):
        """ Add point to the grid """
        px,py,pz = self.find_cell(cd)
                 
        self.index.append(list(cd))
        self.g[px][py][pz].append(self.len_index)

        self.len_index += 1
        self.n_nodes += 1 

    def fill_g(self, points):
        """ Add cloud of points to the grid """
        for point in points:
            self.add_point(point)

    def query(self, q, k=2):
        """ Nearest neighbor function """
        nn = []
        qx,qy,qz = q
        px,py,pz = self.find_cell(q)

        o = self.o
        l = self.l
        n = self.n

        b = min([min(abs(qx - o - px*l), abs(qx - o - min(px+1,n)*l)),
                 min(abs(qy - o - py*l), abs(qy - o - min(py+1,n)*l)),
                 min(abs(qz - o - pz*l), abs(qy - o - min(py+1,n)*l))])

        i = 0
        nn = []
        while len(nn)<k:
            candidates = self.get_candidates(px,py,pz,i)            
            if len(candidates):
                nn = self.rank_by_dist(q, candidates)
            i += 1
        
        return nn[:k]  

    def get_candidates(self, px,py,pz, i):
        """ Get all candidates from cells i-away from """
        if not i:
            cells = [[px,py,pz]]
        else:
            cx = set([min(max(0,x),self.n) for x in range(px-i,px+i)])
            cy = set([min(max(0,y),self.n) for y in range(py-i,py+i)])
            cz = set([min(max(0,z),self.n) for z in range(pz-i,pz+i)])
            cells = [[x,y,z] for x,y,z in product(cx,cy,cz)]
        
        candidates = []
        for x,y,z in cells:
            cell_points = [p for p in self.g[x][y][z]]
            candidates += cell_points

        return candidates

    def rank_by_dist(self, q, c):
        """ Given a candidate set, compute distance to query """
        q = np.expand_dims(q, axis=0)
        c_p = np.array([self.index[i] for i in c])
        dists = cdist(q,c_p)[0]
        c_sorted = [c[i] for i in np.argsort(dists)]
        return c_sorted

    
if __name__=="__main__":
    point_cloud = random_point(d=(5000,3))
    gug = GrowingUniformGrid(point_cloud)

    point = [0,0,0]

    from time import time
    t0 = time()

    print(f"Neighbors to point {point}")
    print([(p,gug.index[p]) for p in gug.query(point)])
    print(f"Computed in {time()-t0} seconds")

