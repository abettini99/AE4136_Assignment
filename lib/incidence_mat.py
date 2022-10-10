#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
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
def construct_tE21_2D(grid: Grid2D) -> tuple[sparse.csr.csr_matrix, sparse.csr.csr_matrix]:

    #### ================== ####
    #### function variables ####
    #### ================== ####
    idx: int; idxLB: int; idxRT: int; N: int; Mt: int; Mn: int
    primal_cells:        npt.NDArray[object]
    primal_faces:        npt.NDArray[object]
    tE21:                sparse.dok.dok_matrix # = sparse.dok_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)
    ntE21:                sparse.dok.dok_matrix
    tE21_tmp:            dict = {}
    ntE21_tmp:            dict = {}

    primal_cells, primal_faces , _  = grid.get_primal()
    N = max(primal_cell.idx for primal_cell in primal_cells)
    Mt = max(primal_face.Tidx for primal_face in primal_faces if primal_face.type == 'internal')
    Mn = max(primal_face.Tidx for primal_face in primal_faces if primal_face.type != 'internal')
    tE21 = sparse.dok_matrix((N+1,Mt+1),dtype=np.int8)
    ntE21 = sparse.dok_matrix((N+1,Mn+1),dtype=np.int8)

    #### ================ ####
    #### E21 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for primal_face in primal_faces:
        idx = primal_face.Tidx
        idxLB = primal_face.cells_idx[0] # idxLB -> dir2D.Lp or dir2D.Bp
        idxRT = primal_face.cells_idx[1] # idxRT -> dir2D.Rp or dir2D.Tp

        if primal_face.type == 'internal':
            tE21_tmp[(idxRT,idx)] = -1  # Note that incoming flux for the right/top cell is inwards, so negative
            tE21_tmp[(idxLB,idx)] = 1   # Note that incoming flux for the left/bottom cell is outwards, so negative
        else:
            if idxRT > -1: ntE21_tmp[(idxRT,idx)] = -1 # Note that all boundary faces are attached to only one virtual/real cell! The other 'cell' is a ghost cell.
            if idxLB > -1: ntE21_tmp[(idxLB,idx)] = 1

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    tE21._update(tE21_tmp) # bypass sanity check
    ntE21._update(ntE21_tmp)
    tE21 = tE21.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    ntE21 = ntE21.tocsr()
    #E21.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    return tE21, ntE21

def construct_E10_2D(grid: Grid2D) -> sparse.csr.csr_matrix:

    #### ================== ####
    #### function variables ####
    #### ================== ####
    idx: int; Nidx: int; Eidx: int; Sidx: int; Widx: int
    dual_faces:         npt.NDArray[object]
    dual_verts:         npt.NDArray[object]
    E10:                sparse.dok.dok_matrix # = sparse.dok_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)
    E10_tmp:            dict = {}

    _, dual_faces , dual_verts  = grid.get_dual()
    N = max(dual_face.Tidx for dual_face in dual_faces if dual_face.type == 'internal')
    M = max(dual_vert.Tidx for dual_vert in dual_verts if dual_vert.type == 'internal')
    E10 = sparse.dok_matrix((N+1,M+1),dtype=np.int8)

    #### ================ ####
    #### E10 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for dual_vert in dual_verts:
        if dual_vert.type == 'internal':
            idx = dual_vert.Tidx
            Nidx = dual_vert.faces_idx[dir2D.N]
            Eidx = dual_vert.faces_idx[dir2D.E]
            Sidx = dual_vert.faces_idx[dir2D.S]
            Widx = dual_vert.faces_idx[dir2D.W]

            if Nidx > -1 and dual_faces[Nidx].type == 'internal': Nidx = dual_faces[Nidx].Tidx; E10_tmp[Nidx,idx] = -1
            if Eidx > -1 and dual_faces[Eidx].type == 'internal': Eidx = dual_faces[Eidx].Tidx; E10_tmp[Eidx,idx] = -1
            if Sidx > -1 and dual_faces[Sidx].type == 'internal': Sidx = dual_faces[Sidx].Tidx; E10_tmp[Sidx,idx] = 1
            if Widx > -1 and dual_faces[Widx].type == 'internal': Widx = dual_faces[Widx].Tidx; E10_tmp[Widx,idx] = 1

    #### ================ ####
    #### E10 finalization ####
    #### ================ ####
    E10._update(E10_tmp) # bypass sanity check
    E10 = E10.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    # E10.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them
    return E10

def construct_E21_2D(grid: Grid2D) -> tuple[sparse.csr.csr_matrix, sparse.csr.csr_matrix]:

    #### ================== ####
    #### function variables ####
    #### ================== ####
    idx: int; Nidx: int; Eidx: int; Sidx: int; Widx: int
    dual_cells:          npt.NDArray[object]
    dual_faces:          npt.NDArray[object]
    tE21:                sparse.dok.dok_matrix
    nE21:                sparse.dok.dok_matrix
    tE21_tmp:            dict = {}
    nE21_tmp:            dict = {}

    dual_cells, dual_faces , _  = grid.get_dual()
    N  = max(dual_cell.idx for dual_cell in dual_cells)
    Mt = max(dual_face.Tidx for dual_face in dual_faces if dual_face.type == 'internal')
    Mn = max(dual_face.Tidx for dual_face in dual_faces if dual_face.type != 'internal')
    tE21 = sparse.dok_matrix((N+1,Mt+1),dtype=np.int8)
    nE21 = sparse.dok_matrix((N+1,Mn+1),dtype=np.int8)

    #### ================ ####
    #### E21 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for dual_face in dual_faces:
        idx = dual_face.Tidx
        idxLT = dual_face.cells_idx[0] # idxLT -> dir2D.Ld or dir2D.Td
        idxRB = dual_face.cells_idx[1] # idxRB -> dir2D.Rd or dir2D.Bd

        if dual_face.type == 'internal':
            tE21_tmp[(idxLT,idx)] = 1  # Note that the circulation contribution for the left/top cell is upwards/rightwards, so positive when considering CCW+ curl
            tE21_tmp[(idxRB,idx)] = -1   # Note that the circulation contribution for the right/bottom cell is downwards/leftwards, so negative when considering CCW+ curl
        else:
            if idxLT > -1: nE21_tmp[(idxLT,idx)] = 1 # Note that all boundary faces are attached to only one real cell! The other 'cell' is a ghost cell.
            if idxRB > -1: nE21_tmp[(idxRB,idx)] = -1

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    tE21._update(tE21_tmp) # bypass sanity check
    nE21._update(nE21_tmp)
    tE21 = tE21.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    nE21 = nE21.tocsr()
    #E21.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    return tE21, nE21


if __name__ == "__main__":
    import matplotlib.pyplot as plt #type: ignore

    N = 3
    M = 3
    grid = Grid2D(N,M,'cosine')
    tE21, ntE21    = construct_tE21_2D(grid)
    E10            = construct_E10_2D(grid)
    E21, nE21      = construct_E21_2D(grid)

    # plt.imshow(E10d.toarray())
    # plt.show()

    # Verify the (negative) transpose of the dual grid is the primal grid operator.
    print(f'tE21 == E10.T : \t{(tE21 == -E10.transpose()).toarray().all()}')

    # Verify that E21d@E10d == 0 when considering only the interior volumes.
    innercell_mask: list = [(np.array(cell.cells_idx) > -1).all() for cell in grid.Dcell_array]
    print(f'E21 @ E10 : \t\t{((E21@E10.toarray())[innercell_mask] == 0).all()} \t(inner volumes only) ')
