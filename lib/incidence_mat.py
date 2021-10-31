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
    N: int                                    = grid.N
    M: int                                    = grid.M
    matPcell_idx:       npt.NDArray[np.int32] = grid.get_all_primal_cells_idx()
    Pconnectivity_idx:  npt.NDArray[np.int32] = grid.get_all_primal_cell2face_connectivity()
    vPface_idx:         npt.NDArray[np.int32] = grid.get_boundary_primal_faces_idx()[1]

    # TODO: Convert to COO matrix
    E21 = sparse.lil_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)

    #### ================ ####
    #### E21 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for i in range(matPcell_idx.shape[0]): # i index := x index
        for j in range(matPcell_idx.shape[1]): # j index := y index
            idx = matPcell_idx[i,j]
            if idx >= 0: # Avoid doing operations in case of spurious cells (idx < 0)
                Nidx = Pconnectivity_idx[dir2D.N][i,j]
                Eidx = Pconnectivity_idx[dir2D.E][i,j]
                Sidx = Pconnectivity_idx[dir2D.S][i,j]
                Widx = Pconnectivity_idx[dir2D.W][i,j]
                E21[idx,Nidx] = 1 if Nidx >= 0 else 0 # Avoid doing operations in case of spurious cells (Nidx, Eidx, ... < 0)
                E21[idx,Eidx] = 1 if Eidx >= 0 else 0
                E21[idx,Sidx] = -1 if Sidx >= 0 else 0
                E21[idx,Widx] = -1 if Widx >= 0 else 0

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    E21 = E21.tocsr() ## lil_matrix good when constructing, but csr better for matrix-vector products
    E21.eliminate_zeros() ## In the case that zeros within the sparse structure exists, then remove them

    tE21 = remove_sparse_rowcol(E21, cols_idx=vPface_idx)
    pE21 = extract_sparse_rowcol(E21, idx=vPface_idx, ext='col')
    return E21, tE21, pE21

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 256
    M = 256
    grid = Grid2D(N,M)
    E21, tE21, pE21 = construct_E21_2D(grid)

    #print(E21)
    plt.imshow(pE21.toarray())
    plt.show()


    # print(grid)
