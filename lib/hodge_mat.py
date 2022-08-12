#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import math as m
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
try:
    from utils import *
    from grid_gen import *
except ModuleNotFoundError:
    from lib.utils import *
    from lib.grid_gen import *

# Functions / Classes
def construct_H11_2D(grid: Grid2D) -> tuple[sparse.csr.csr_matrix, sparse.csr.csr_matrix]:
    #### ================== ####
    #### function variables ####
    #### ================== ####
    Pidx: int; Didx: int
    primal_faces:        npt.NDArray[object]
    dual_faces:          npt.NDArray[object]
    primal_vertices:     npt.NDArray[object]
    dual_vertices:       npt.NDArray[object]
    H1t1:                sparse.dok.dok_matrix # = sparse.dok_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)
    H1t1_tmp:            dict = {}
    Ht11:                sparse.dok.dok_matrix # = sparse.dok_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)
    Ht11_tmp:            dict = {}

    _, primal_faces, primal_vertices  = grid.get_primal()
    _, dual_faces, dual_vertices      = grid.get_dual()
    N = max(dual_face.Tidx for dual_face in dual_faces if dual_face.type == 'internal')
    M = max(primal_face.Tidx for primal_face in primal_faces if primal_face.type == 'internal')
    H1t1 = sparse.dok_matrix((N+1,M+1),dtype=np.float64)
    Ht11 = sparse.dok_matrix((M+1,N+1),dtype=np.float64)

    for primal_face in primal_faces:
        if primal_face.type == 'internal':

            # Grab temporary indices
            Pidx = primal_face.idx
            Didx = primal_face.Midx

            # Grab face indices
            idxp0 = primal_faces[Pidx].vertices_idx[0]
            idxp1 = primal_faces[Pidx].vertices_idx[1]
            idxd0 = dual_faces[Didx].vertices_idx[0]
            idxd1 = dual_faces[Didx].vertices_idx[1]

            # Grab true indices
            Pidx = primal_face.Tidx
            Didx = dual_faces[Didx].Tidx

            # Grab information
            xp0,yp0 = primal_vertices[idxp0].coordinates[dir2D.x], primal_vertices[idxp0].coordinates[dir2D.y]
            xp1,yp1 = primal_vertices[idxp1].coordinates[dir2D.x], primal_vertices[idxp1].coordinates[dir2D.y]
            xd0,yd0 = dual_vertices[idxd0].coordinates[dir2D.x], dual_vertices[idxd0].coordinates[dir2D.y]
            xd1,yd1 = dual_vertices[idxd1].coordinates[dir2D.x], dual_vertices[idxd1].coordinates[dir2D.y]

            # Calculate spacing
            h_primal = m.sqrt( (xp0-xp1)**2 + (yp0-yp1)**2 )
            h_dual = m.sqrt( (xd0-xd1)**2 + (yd0-yd1)**2 )

            # Construct Hodge
            H1t1_tmp[(Didx,Pidx)] = h_dual/h_primal
            Ht11_tmp[(Pidx,Didx)] = h_primal/h_dual

    #### ================ ####
    #### H11 finalization ####
    #### ================ ####
    H1t1._update(H1t1_tmp) # bypass sanity check
    H1t1 = H1t1.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    Ht11._update(Ht11_tmp) # bypass sanity check
    Ht11 = Ht11.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    # H1t1.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    return H1t1, Ht11


# Functions / Classes
def construct_Ht02_2D(grid: Grid2D) -> sparse.csr.csr_matrix:
    #### ================== ####
    #### function variables ####
    #### ================== ####
    Pidx: int; Didx: int
    primal_vertices:     npt.NDArray[object]
    dual_cells:          npt.NDArray[object]
    Ht02:                sparse.dok.dok_matrix # = sparse.dok_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)
    Ht02_tmp:            dict = {}

    _, _, primal_vertices  = grid.get_primal()
    dual_cells, _, _       = grid.get_dual()
    N = max(primal_vertice.Tidx for primal_vertice in primal_vertices if primal_vertice.type == 'internal')
    M = max(dual_cell.Tidx for dual_cell in dual_cells if dual_cell.type == 'internal')
    # H0t2 = sparse.dok_matrix((N+1,M+1),dtype=np.float64)
    Ht02 = sparse.dok_matrix((N+1,M+1),dtype=np.float64)

    for dual_cell in dual_cells:
        if dual_cell.type == 'internal':

            # Grab temporary indices
            Pidx = dual_cell.Midx
            Didx = dual_cell.idx

            # Grab true indices
            Pidx = primal_vertices[Pidx].Tidx
            Didx = dual_cell.Tidx

            # Construct Hodge
            Ht02_tmp[(Pidx,Didx)] = 1/dual_cell.volume

    #### ================= ####
    #### Ht02 finalization ####
    #### ================= ####
    Ht02._update(Ht02_tmp) # bypass sanity check
    Ht02 = Ht02.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    # Ht02.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    return Ht02

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from incidence_mat import *

    N = 3
    M = 3

    grid = Grid2D(N,M,'cosine')

    tE21, ntE21     = construct_tE21_2D(grid)
    E10             = construct_E10_2D(grid)
    E21, nE21       = construct_E21_2D(grid)
    H1t1,Ht11       = construct_H11_2D(grid)
    Ht02            = construct_Ht02_2D(grid)

    # plt.imshow(Ht02.toarray())
    # plt.show()

    # plt.imshow(tE21@Ht11@E10.toarray())
    # plt.show()

    # plt.imshow(H1t1@E21.transpose()@Ht02.toarray())
    # plt.show()
