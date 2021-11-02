#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
from utils import *
from grid_gen import *

# Functions / Classes
def construct_E21_2D(grid: Grid2D) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Generates the E21 matrix, used for discrete flux summation.
       Flux summation is done on the primal grid, with cell-outward flux
       being positive flux, and flux arrows going left to right, and bottom to
       top.

       Parameters
       ----------
       grid : Grid2D
            Generated structured 2D grid.

       Returns
       -------
       E21 : sparse.csr_matrix
            Sparse E21 matrix (discrete flux summation for a cell, outgoing
            positive, flux: left to right, bottom to top)

       tE21 : sparse.csr_matrix
            Sparse truncated E21 matrix (discrete flux summation for a cell,
            outgoing positive, flux: left to right, bottom to top). Boundary edge
            contributions removed as they are prescribed.

       pE21 : sparse.csr_matrix
            Sparse prescribed E21 matrix (discrete flux summation for a cell,
            outgoing positive, flux: left to right, bottom to top). Boundary edge
            contributions to the flux of each cell effectively. Mostly zero.

       Examples
       --------
       grid            = Grid2D(64,32)
       E21, tE21, pE21 = construct_E21_2D(grid)
       """

    #### ================== ####
    #### function variables ####
    #### ================== ####
    matPcell_idx:       npt.NDArray[np.int32] = grid.get_all_primal_cells_idx()
    Pconnectivity_idx:  npt.NDArray[np.int32] = grid.get_all_primal_cell2face_connectivity()
    bPface_idx:         npt.NDArray[np.int32] = grid.get_boundary_primal_faces_idx()[1]

    # TODO: Convert to COO matrix
    E21 = sparse.lil_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+2),dtype=np.int8)

    #### ================ ####
    #### E21 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for i in range(matPcell_idx.shape[0]): # i index := x index
        for j in range(matPcell_idx.shape[1]): # j index := y index
            idx = matPcell_idx[i,j]
            if idx >= 0: # Avoid doing operations in case of spurious cells (idx < 0)
                Nidx: int = Pconnectivity_idx[dir2D.N][i,j]
                Eidx: int = Pconnectivity_idx[dir2D.E][i,j]
                Sidx: int = Pconnectivity_idx[dir2D.S][i,j]
                Widx: int = Pconnectivity_idx[dir2D.W][i,j]
                E21[idx,Nidx] = 1 if Nidx >= 0 else 0 # Avoid doing operations in case of spurious faces (Nidx, Eidx, ... < 0)
                E21[idx,Eidx] = 1 if Eidx >= 0 else 0
                E21[idx,Sidx] = -1 if Sidx >= 0 else 0
                E21[idx,Widx] = -1 if Widx >= 0 else 0
    E21 = E21[:,:-1] # I have no clue why I need to do this, but for some reason this fixes issues with the last row, last column entry.
    # TODO: fix the E21 bug.

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    E21 = E21.tocsr() # lil_matrix good when constructing, but csr better for matrix-vector products
    E21.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    tE21 = remove_sparse_rowcol(E21, cols_idx=bPface_idx)
    pE21 = extract_sparse_rowcol(E21, idx=bPface_idx, ext='col')
    return E21, tE21, pE21

def construct_E10_2D(grid: Grid2D) -> sparse.csr_matrix:
    """Generates the E10 matrix, used for discrete gradient estimations.
       Gradient estimation is done on the dual grid, with sink-like vertices,
       edge orientation left to right, bottom to top.

       Parameters
       ----------
       grid : Grid2D
            Generated structured 2D grid.

       Returns
       -------
       E10 : sparse.csr_matrix
            Sparse E10 matrix (discrete gradient, sink-like vertices, edge
            orientation left to right, bottom to top)

       Examples
       --------
       grid = Grid2D(64,32)
       E10  = construct_E10_2D(grid)
       """

    #### ================== ####
    #### function variables ####
    #### ================== ####
    matDvert_idx:       npt.NDArray[np.int32] = grid.get_all_dual_verts_idx()
    Dconnectivity_idx:  npt.NDArray[np.int32] = grid.get_all_dual_vert2face_connectivity()
    bDface_idx:         npt.NDArray[np.int32] = grid.get_boundary_dual_faces_idx()[1]

    # TODO: Convert to COO matrix, note that E10 is initialized with an extra row, see comment below.
    E10 = sparse.lil_matrix((Dconnectivity_idx.max()+2,matDvert_idx.max()+1),dtype=np.int8)

    #### ================ ####
    #### E10 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for i in range(matDvert_idx.shape[0]): # i index := x index
        for j in range(matDvert_idx.shape[1]): # j index := y index
            idx = matDvert_idx[i,j]
            if idx >= 0: # Avoid doing operations in case of spurious vertex (idx < 0)
                Nidx: int = Dconnectivity_idx[dir2D.N][i,j]
                Eidx: int = Dconnectivity_idx[dir2D.E][i,j]
                Sidx: int = Dconnectivity_idx[dir2D.S][i,j]
                Widx: int = Dconnectivity_idx[dir2D.W][i,j]
                E10[Nidx,idx] = -1 if Nidx >= 0 else 0 # Avoid doing operations in case of spurious faces (Nidx, Eidx, ... < 0)
                E10[Eidx,idx] = -1 if Eidx >= 0 else 0
                E10[Sidx,idx] = 1 if Sidx >= 0 else 0
                E10[Widx,idx] = 1 if Widx >= 0 else 0
    E10 = E10[:-1,:] # I have no clue why I need to do this, but for some reason this fixes issues with the last row, last column entry.
    # TODO: fix the E10 bug.

    #### ================ ####
    #### E10 finalization ####
    #### ================ ####
    E10 = E10.tocsr() # lil_matrix good when constructing, but csr better for matrix-vector products
    E10.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them
    tE10 = remove_sparse_rowcol(E10, rows_idx=bDface_idx)
    #print(Dconnectivity_idx[dir2D.S])
    return tE10

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 3
    M = 3
    grid = Grid2D(N,M)
    E21, tE21, pE21 = construct_E21_2D(grid)
    E10             = construct_E10_2D(grid)

    print(tE21 != -E10.transpose())

    # plt.imshow(E10.toarray()) # # TODO: Maybe something wrong here? Need to troubleshoot
    # plt.show()
    plt.imshow(-E10.transpose().toarray())
    plt.show()
    plt.imshow(tE21.toarray())
    plt.show()


    # print(grid)
