#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
from utils import *
from grid_gen import *

# Functions / Classes
def construct_E21_2D(grid: Grid2D) -> sparse.csr.csr_matrix:
    """Generates a two dimensional grid provided the number of interior cells in
       the x-y direction.

       Parameters
       ----------
       grid : Grid2D
            Generated structured 2D grid.

       Returns
       -------
       E21 : sparse.csr.csr_matrix
            Sparse E21 matrix (discrete flux summation for a cell, outgoing
            positive)

       Examples
       --------
       grid = Grid2D(64,32)
       E21  = construct_E21_2D(grid)
       """

    #### ================== ####
    #### function variables ####
    #### ================== ####
    N: int                                  = grid.N
    M: int                                  = grid.M
    matcell_idx:      npt.NDArray[np.int32] = grid.get_all_cells_idx()
    connectivity_idx: npt.NDArray[np.int32] = grid.get_all_cell2face_connectivity()

    E21 = sparse.lil_matrix((matcell_idx.max()+1, connectivity_idx.max()+1),dtype=np.int8)

    #### ================ ####
    #### E21 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for i in range(matcell_idx.shape[0]): # i index := x index
        for j in range(matcell_idx.shape[1]): # j index := y index
            idx = matcell_idx[i,j]
            if idx >= 0: # Avoid doing operations in case of spurious cells (idx < 0)
                Nidx = connectivity_idx[dir2D.N][i,j]
                Eidx = connectivity_idx[dir2D.E][i,j]
                Sidx = connectivity_idx[dir2D.S][i,j]
                Widx = connectivity_idx[dir2D.W][i,j]
                E21[idx,Nidx] = 1 if Nidx >= 0 else 0 # Avoid doing operations in case of spurious cells (Nidx, Eidx, ... < 0)
                E21[idx,Eidx] = 1 if Eidx >= 0 else 0
                E21[idx,Sidx] = -1 if Sidx >= 0 else 0
                E21[idx,Widx] = -1 if Widx >= 0 else 0

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    E21 = E21.tocsr() ## lil_matrix good when constructing, but csr better for matrix-vector products
    E21.eliminate_zeros() ## In the case that zeros within the sparse structure exists, then remove them

    # TODO: construct tE21 and give it as second output.
    return E21

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 256
    M = 256
    grid = Grid2D(N,M)
    E21 = construct_E21_2D(grid)
    print(E21)
    plt.spy(E21)
    plt.show()


    # print(grid)
