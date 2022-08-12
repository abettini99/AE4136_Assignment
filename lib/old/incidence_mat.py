#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
from utils import *
from grid_gen import *

# Functions / Classes
def construct_primal_E21_2D(grid: Grid2D) -> tuple[sparse.csr.csr_matrix, sparse.csr.csr_matrix, sparse.csr.csr_matrix]:
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
       E21 : sparse.csr.csr_matrix
            Sparse E21 matrix (discrete flux summation for a cell, outgoing
            positive, flux: left to right, bottom to top)

       tE21 : sparse.csr.csr_matrix
            Sparse truncated E21 matrix (discrete flux summation for a cell,
            outgoing positive, flux: left to right, bottom to top). Boundary edge
            contributions removed as they are prescribed.

       pE21 : sparse.csr.csr_matrix
            Sparse prescribed E21 matrix (discrete flux summation for a cell,
            outgoing positive, flux: left to right, bottom to top). Boundary edge
            contributions to the flux of each cell effectively. Mostly zero.

       Examples
       --------
       grid            = Grid2D(64,32)
       E21, tE21, pE21 = construct_primal_E21_2D(grid)
       """

    #### ================== ####
    #### function variables ####
    #### ================== ####
    matPcell_idx:       npt.NDArray[np.int32] = grid.get_all_primal_cells_idx()
    Pconnectivity_idx:  npt.NDArray[np.int32] = grid.get_all_primal_cell2face_connectivity()
    bPface_idx:         npt.NDArray[np.int32] = grid.get_boundary_primal_faces_idx()[1]
    E21:                sparse.dok.dok_matrix = sparse.dok_matrix((matPcell_idx.max()+1, Pconnectivity_idx.max()+1),dtype=np.int8)
    E21_tmp:            dict                  = {}
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
                if Nidx >= 0: E21_tmp[(idx,Nidx)] = 1 # Avoid doing operations in case of spurious faces (Nidx, Eidx, ... < 0)
                if Eidx >= 0: E21_tmp[(idx,Eidx)] = 1
                if Sidx >= 0: E21_tmp[(idx,Sidx)] = -1
                if Widx >= 0: E21_tmp[(idx,Widx)] = -1

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    E21._update(E21_tmp) # bypass sanity check
    E21 = E21.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    # E21.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    tE21 = remove_sparse_rowcol(E21, cols_idx=bPface_idx)
    pE21 = extract_sparse_rowcol(E21, idx=bPface_idx, ext='col')
    return E21, tE21, pE21

def construct_dual_E10_2D(grid: Grid2D) -> sparse.csr.csr_matrix:
    """Generates the E10 matrix, used for discrete gradient estimations.
       Gradient estimation is done on the dual grid, with sink-like vertices,
       edge orientation left to right, bottom to top.

       Parameters
       ----------
       grid : Grid2D
            Generated structured 2D grid.

       Returns
       -------
       E10 : sparse.csr.csr_matrix
            Sparse E10 matrix (discrete gradient, sink-like vertices, edge
            orientation left to right, bottom to top)

       Examples
       --------
       grid = Grid2D(64,32)
       E10  = construct_dual_E10_2D(grid)
       """

    #### ================== ####
    #### function variables ####
    #### ================== ####
    matDvert_idx:       npt.NDArray[np.int32] = grid.get_all_dual_verts_idx()
    Dconnectivity_idx:  npt.NDArray[np.int32] = grid.get_all_dual_vert2face_connectivity()
    bDface_idx:         npt.NDArray[np.int32] = grid.get_boundary_dual_faces_idx()[1]
    E10:                sparse.dok.dok_matrix = sparse.dok_matrix((Dconnectivity_idx.max()+1,matDvert_idx.max()+1),dtype=np.int8)
    E10_tmp:            dict                  = {}

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
                if Nidx >= 0: E10_tmp[(Nidx,idx)] = -1 # Avoid doing operations in case of spurious faces (Nidx, Eidx, ... < 0)
                if Eidx >= 0: E10_tmp[(Eidx,idx)] = -1
                if Sidx >= 0: E10_tmp[(Sidx,idx)] = 1
                if Widx >= 0: E10_tmp[(Widx,idx)] = 1

    #### ================ ####
    #### E10 finalization ####
    #### ================ ####
    E10._update(E10_tmp) # bypass sanity check
    E10 = E10.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    # E10.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them
    tE10 = remove_sparse_rowcol(E10, rows_idx=bDface_idx)
    #print(Dconnectivity_idx[dir2D.S])
    return tE10

def construct_dual_E21_2D(grid: Grid2D) -> tuple[sparse.csr.csr_matrix, sparse.csr.csr_matrix, sparse.csr.csr_matrix]:
    """Generates the E21 matrix, used for discrete curl summation.
       Curl summation is done on the dual grid, with counter-clockwise direction
       being positive curl, and curl arrows going left to right, and bottom to
       top.

       Parameters
       ----------
       grid : Grid2D
            Generated structured 2D grid.

       Returns
       -------
       E21 : sparse.csr.csr_matrix
            Sparse E21 matrix (discrete curl summation for a cell, counter-
            clockwise positive, curl arrows: left to right, bottom to top)

       tE21 : sparse.csr.csr_matrix
            Sparse truncated E21 matrix (discrete flux summation for a cell,
            outgoing positive, flux: left to right, bottom to top). Boundary edge
            contributions removed as they are prescribed.

       pE21 : sparse.csr.csr_matrix
            Sparse prescribed E21 matrix (discrete flux summation for a cell,
            outgoing positive, flux: left to right, bottom to top). Boundary edge
            contributions to the flux of each cell effectively. Mostly zero.

       Examples
       --------
       grid            = Grid2D(64,32)
       E21, tE21, pE21 = construct_dual_E21_2D(grid)
       """

    #### ================== ####
    #### function variables ####
    #### ================== ####
    matDcell_idx:       npt.NDArray[np.int32] = grid.get_all_dual_cells_idx()
    Dconnectivity_idx:  npt.NDArray[np.int32] = grid.get_all_dual_cell2face_connectivity()
    bDface_idx:         npt.NDArray[np.int32] = grid.get_boundary_dual_faces_idx()[1]
    E21:                sparse.dok.dok_matrix = sparse.dok_matrix((matDcell_idx.max()+1, Dconnectivity_idx.max()+1),dtype=np.int8)
    E21_tmp:            dict                  = {}
    #### ================ ####
    #### E21 construction ####
    #### ================ ####
    ## Loop through the 2D grid
    for i in range(matDcell_idx.shape[0]): # i index := x index
        for j in range(matDcell_idx.shape[1]): # j index := y index
            idx = matDcell_idx[i,j]
            if idx >= 0: # Avoid doing operations in case of spurious cells (idx < 0)
                Nidx: int = Dconnectivity_idx[dir2D.N][i,j]
                Eidx: int = Dconnectivity_idx[dir2D.E][i,j]
                Sidx: int = Dconnectivity_idx[dir2D.S][i,j]
                Widx: int = Dconnectivity_idx[dir2D.W][i,j]
                if Nidx >= 0: E21_tmp[(idx,Nidx)] = -1 # Avoid doing operations in case of spurious faces (Nidx, Eidx, ... < 0)
                if Eidx >= 0: E21_tmp[(idx,Eidx)] = 1
                if Sidx >= 0: E21_tmp[(idx,Sidx)] = 1
                if Widx >= 0: E21_tmp[(idx,Widx)] = -1

    #### ================ ####
    #### E21 finalization ####
    #### ================ ####
    E21._update(E21_tmp) # bypass sanity check
    E21 = E21.tocsr() # dok_matrix good when constructing, but csr better for matrix-vector products
    # E21.eliminate_zeros() # In the case that zeros within the sparse structure exists, then remove them

    tE21 = remove_sparse_rowcol(E21, cols_idx=bDface_idx)
    pE21 = extract_sparse_rowcol(E21, idx=bDface_idx, ext='col')
    return E21, tE21, pE21


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 5
    M = 5
    grid = Grid2D(N,M)
    _, E21p, pE21p  = construct_primal_E21_2D(grid)
    E10d            = construct_dual_E10_2D(grid)
    _, E21d, pE21d  = construct_dual_E21_2D(grid)

    # Verify the (negative) transpose of the dual grid is the primal grid operator.
    print(E21p != -E10d.transpose())

    # Verify the curl of the gradient is zero. Note that only the inside cells are considered since prescribing values means that the curl of the gradient
    # is not necessarily zero on those cells.
    innercell_idx = grid.get_inner_dual_cells_idx()
    iE21d = remove_sparse_rowcol(E21d, rows_idx=innercell_idx)
    print(iE21d@E10d != 0)

    # Verify that the divergence of the curl is zero. Note that only the inside cells are considered since prescribing values means that the curl
    # is not necessarily zero on those cells.
    print(E21p@iE21d.transpose() != 0)

    # plt.imshow(E10d.toarray()) # TODO: Maybe something wrong here? Need to troubleshoot
    # plt.show()
    # plt.imshow(-E10d.transpose().toarray())
    # plt.show()
    # plt.imshow(pE21p.toarray())
    # plt.show()
    # plt.imshow((iE21d@E10d).toarray())
    plt.imshow((E10d).toarray())
    plt.show()
    # plt.imshow((E21p@iE21d.transpose()).toarray())
    # plt.show()

    # print(grid)
