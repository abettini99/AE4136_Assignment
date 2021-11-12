#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from utils import *

# Functions / Classes
@dataclass() # Not super necessary to have this decorator for this to work, but it is still nice to have.
class Grid2D:
    """Generates a two dimensional grid (primal and dual) provided the
       number of interior cells in the x-y direction.

       Parameters
       ----------
       N : int
            Number of cells in the x-direction.
       M : int
            Number of cells in the y-direction.

       Class Members
       -------------
       get_internal_primal_cells_idx() -> array_like ( [N, M] )
           returns indices of all interior primal cells in grid form.

       get_boundary_primal_cells_idx() -> list[ array_like ] ( [4 -> [x]] ), array_like
           returns indices of all boundary primal cells in N,E,S,W format in grid
           form. Also returns the list in a flattened, sorted array.

       get_boundary_primal_faces_idx() -> list[ array_like ] ( [4 -> [x]] ), array_like
           returns indices of all boundary primal faces in N,E,S,W format in grid
           form. Also returns the list in a flattened, sorted array.

       get_all_primal_cells_idx() -> array_like ( [N+2, M+2] )
           returns indices of all primal cells in grid form.

       get_all_primal_faces_idx() -> list[ array_like ] ( [2 -> [x,x]] )
           returns indices of all primal faces in V, H format and in grid form.

       get_all_primal_verts_idx() -> array_like ( [N+1, M+1] )
           returns indices of all primal vertices in grid form.

       get_all_primal_cell2face_connectivity() -> array_like ( [4, N+2, M+2] )
           returns indices of all primal faces connected to each primal cell in
           grid form. The first index refers to the direction (N E S W), and the
           other two indices refer to the primal cell index in the x-y frame.

       get_all_primal_vert2face_connectivity() -> array_like ( [4, N+1, M+1] )
           returns indices of all primal faces connected to each primal vertex
           in grid form. The first index refers to the direction (N E S W), and
           the other two indices refer to the primal vertex index in the x-y
           frame.

       get_internal_primal_cell2vert_connectivity() -> array_like ( [4, N, M] )
           returns indices of all primal vertices connected to each interior
           primal cell in grid form. The first index refers to the direction
           (SW SE NE NW), and the other two indices refer to the primal cell
           index in the x-y frame.

       get_internal_primal_cell2face_connectivity() -> array_like ( [4, N, M] )
           returns indices of all primal faces connected to each internal primal
           cell in grid form. The first index refers to the direction (N E S W),
           and the other two indices refer to the primal cell index in the x-y
           frame.

           --

       get_boundary_dual_faces_idx() -> list[ array_like ] ( [4 -> [x]] ), array_like
           returns indices of all boundary dual faces in N,E,S,W format in grid
           form. Also returns the list in a flattened, sorted array.

       get_all_dual_cells_idx() -> array_like ( [N+2, M+2] )
           returns indices of all dual cells in grid form.

       get_all_dual_faces_idx() -> list[ array_like ] ( [2 -> [x,x]] )
           returns indices of all dual faces in V, H format and in grid form.

       get_all_dual_verts_idx() -> array_like ( [N+1, M+1] )
           returns indices of all dual vertices in grid form.

       get_all_dual_cell2face_connectivity() -> array_like ( [4, N+2, M+2] )
           returns indices of all dual faces connected to each dual cell in
           grid form. The first index refers to the direction (N E S W), and the
           other two indices refer to the dual cell index in the x-y frame.

       get_all_dual_vert2face_connectivity() -> array_like ( [4, N+1, M+1] )
           returns indices of all dual faces connected to each dual vertex in
           grid form. The first index refers to the direction (N E S W), and
           the other two indices refer to the dual vertex index in the x-y
           frame.

       get_internal_dual_cell2vert_connectivity() -> array_like ( [4, N+1, M+1] )
           returns indices of all dual vertices connected to each interior dual
           cell in grid form. The first index refers to the direction (SW SE NE
           NW), and the other two indices refer to the dual cell index in the
           x-y frame.

       Notes
       -----
       An index of -1 refers to a spurious datapoint (e.g. face or cell).

       Examples
       --------
       grid = Grid2D(64,32)
       int_primal_cell2face_connectivity = grid.get_internal_primal_cell2face_connectivity()
       """

    #### ============ ####
    #### Class inputs ####
    #### ============ ####
    N: int  ## Number of cells in the x-direction
    M: int  ## Number of cells in the y-direction

    #### =================== ####
    #### Post-init variables ####
    #### =================== ####
    # Abbreviations
    # P     - primal
    # D     - dual
    # b     - boundary
    # mat   - matrix

    matPcell_idx:       npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+2, M+2]      // primal cell numbering, real, boundary, and spurious cells
    matPface_idx:       list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [2 -> [x,x]]    // primal face numbering, real, boundary, and spurious faces
    matPvert_idx:       npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+1, M+1]      // primal vertices numbering, real vertices
    matPcellface_idx:   npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+2, M+2]   // primal cell-face connectivity, real, boundary, and spurious faces
    matPvertface_idx:   npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+1, M+1]   // primal vertex-face connectivity, real and spurious faces
    matPcellvert_idx:   npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N, M]       // primal cell-vertex connectivity, real cells
    bPcells_idx:        list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // primal boundary cell indices
    bPface_idx:         list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // primal boundary cell indices

    # # TODO: update comments
    matDcell_idx:       npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+1, M+1]      // dual cell numbering, real cells
    matDface_idx:       list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [2 -> [x,x]]    // dual face numbering, real and spurious faces
    matDvert_idx:       npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+2, M+2]      // dual vertex numbering, real and spurious vertex
    matDcellface_idx:   npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+2, M+2]   // dual cell-face connectivity, real and spurious faces
    matDvertface_idx:   npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+1, M+1]   // dual vertex-face connectivity, real and spurious faces
    matDcellvert_idx:   npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+1, M+1]   // dual cell-vertex connectivity, real cells, disparity between primal and this is that N+1 and M+1 dual cells exist, not N, M.
    bDface_idx:         list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // primal boundary cell indices

    def __post_init__(self) -> None:
        #### ======================== ####
        #### mypy variable annotation ####
        #### ======================== ####
        matPcell_idx:       npt.NDArray[np.int32]
        matPface_idx:       list[npt.NDArray[np.int32]]
        matPvert_idx:       npt.NDArray[np.int32]
        matPcellface_idx:   npt.NDArray[np.int32]
        matPvertface_idx:   npt.NDArray[np.int32]
        matPcellvert_idx:   npt.NDArray[np.int32]
        matPfaceH_idx:      npt.NDArray[np.int32]
        matPfaceV_idx:      npt.NDArray[np.int32]
        matPcellmask_idx:   npt.NDArray[np.bool_]
        matPfacemaskH_idx:  npt.NDArray[np.bool_]
        matPfacemaskV_idx:  npt.NDArray[np.bool_]

        matDcell_idx:       npt.NDArray[np.int32]
        matDface_idx:       list[npt.NDArray[np.int32]]
        matDvert_idx:       npt.NDArray[np.int32]
        matDcellface_idx:   npt.NDArray[np.int32]
        matDvertface_idx:   npt.NDArray[np.int32]
        matDcellvert_idx:   npt.NDArray[np.int32]
        matDfaceH_idx:      npt.NDArray[np.int32]
        matDfaceV_idx:      npt.NDArray[np.int32]
        matDcellmask_idx:   npt.NDArray[np.bool_]
        matDfacemaskH_idx:  npt.NDArray[np.bool_]
        matDfacemaskV_idx:  npt.NDArray[np.bool_]

        bPcells_idx:        list[npt.NDArray[np.int32]]
        bPface_idx:         list[npt.NDArray[np.int32]]
        bDface_idx:         list[npt.NDArray[np.int32]]
        tmp:                npt.NDArray[np.int32] = np.asarray(0, dtype=np.int32)

        #### ================================= ####
        #### primal cell-level data generation ####
        #### ================================= ####
        ## Note that the center of the primal cell is the center.
        matPcell_idx      = np.arange( (self.N+2)*(self.M+2), dtype=np.int32 )                  ## Create full grid index (real, boundary, spurious cells)
        matPcell_idx      = matPcell_idx.reshape( (self.N+2,self.M+2) )                         ## Reshape it into a grid
        matPcell_idx[0,0] = matPcell_idx[-1,0] = matPcell_idx[0,-1] = matPcell_idx[-1,-1] = -1  ## Set spurious index to -1 -- always the corner cases
        matPcellmask_idx  = matPcell_idx >= 0                                                   ## Mask spurious cells

        matPcell_idx[matPcellmask_idx]           -= 1                                           ## Fix grid cell numbering such that index is continuous after disregarding spurious cells
        matPcell_idx[1:][matPcellmask_idx[1:]]   -= 1
        matPcell_idx[-1:][matPcellmask_idx[-1:]] -= 1

        #### ================================= ####
        #### primal face-level data generation ####
        #### ================================= ####
        matPfaceV_idx                     = np.arange( (self.N+2)*(self.M+3), dtype=np.int32 )   ## Create full grid index for vertical faces
        matPfaceV_idx                     = matPfaceV_idx.reshape( (self.N+2,self.M+3) )         ## Reshape it into grid
        matPfaceV_idx[0,:]                = matPfaceV_idx[-1,:] = -1                             ## Set spurious index to -1
        matPfacemaskV_idx                 = matPfaceV_idx >= 0                                   ## Mask spurious cells
        matPfaceV_idx[matPfacemaskV_idx] -= self.M+3

        matPfaceH_idx                     = np.arange( (self.N+3)*(self.M+2), dtype=np.int32 )   ## Create full grid index for vertical faces
        matPfaceH_idx                     = matPfaceH_idx.reshape( (self.N+3,self.M+2) )         ## Reshape it into grid
        matPfaceH_idx[:,0]                = matPfaceH_idx[:,-1] = -1                             ## Set spurious index to -1
        matPfacemaskH_idx                 = matPfaceH_idx >= 0                                   ## Mask spurious cells
        matPfaceH_idx[matPfacemaskH_idx] += matPfaceV_idx.max()
        for i in range(1,self.N+3):
            matPfaceH_idx[i][matPfacemaskH_idx[i]] -= 2*i

        matPface_idx          = [tmp, tmp]
        matPface_idx[dir2D.V] = matPfaceV_idx
        matPface_idx[dir2D.H] = matPfaceH_idx

        #### =================================== ####
        #### primal vertex-level data generation ####
        #### =================================== ####
        ## Very easy to do since all vertices that are considered are real.
        # TODO: Most likely wrong at the moment, probably requires boundary cells as well, even in primal form
        matPvert_idx                    = np.arange( (self.N+1)*(self.M+1), dtype=np.int32 )    ## Create full grid index for vertices
        matPvert_idx                    = matPvert_idx.reshape( (self.N+1,self.M+1) )           ## Reshape it into grid

        #### =================================== ####
        #### primal connectivity data generation ####
        #### =================================== ####
        ## Generate primal cell-face connectivity data
        matPcellface_idx               = np.empty( (4,(self.N+2),(self.M+2)), dtype=np.int32 )
        matPcellface_idx[dir2D.N,:,:]  = matPface_idx[dir2D.H][1:,:];
        matPcellface_idx[dir2D.E,:,:]  = matPface_idx[dir2D.V][:,1:];
        matPcellface_idx[dir2D.S,:,:]  = matPface_idx[dir2D.H][:-1,:];
        matPcellface_idx[dir2D.W,:,:]  = matPface_idx[dir2D.V][:,:-1];

        ## Generate primal vertex-face connectivity data
        matPvertface_idx               = np.empty( (4,(self.N+1),(self.M+1)), dtype=np.int32 )
        matPvertface_idx[dir2D.N,:,:]  = matPface_idx[dir2D.V][1:,1:-1];
        matPvertface_idx[dir2D.E,:,:]  = matPface_idx[dir2D.H][1:-1,1:];
        matPvertface_idx[dir2D.S,:,:]  = matPface_idx[dir2D.V][:-1,1:-1];
        matPvertface_idx[dir2D.W,:,:]  = matPface_idx[dir2D.H][1:-1,:-1];

        ## Generate primal cell-vertex connectivity data
        matPcellvert_idx               = np.empty( (4,(self.N),(self.M)), dtype=np.int32 )
        matPcellvert_idx[dir2D.SW,:,:] = matPvert_idx[:-1,:-1];
        matPcellvert_idx[dir2D.SE,:,:] = matPvert_idx[1: ,:-1];
        matPcellvert_idx[dir2D.NE,:,:] = matPvert_idx[1: , 1:];
        matPcellvert_idx[dir2D.NW,:,:] = matPvert_idx[1: ,:-1];

        #### ================================= ####
        #### dual vertex-level data generation ####
        #### ================================= ####
        ## The vertices of the dual grid coincide with the center of the primal cells.
        matDvert_idx = np.copy(matPcell_idx)

        #### =============================== ####
        #### dual face-level data generation ####
        #### =============================== ####
        ## The face of the dual grid cell **nearly** coincide with the **numbering** of the primal faces
        ## The main difference being that the boundary faces do not exist.
        ## Additionally, note that the x-derivatives given first, so the horizontal faces come first!
        matDface_idx  = [tmp, tmp]

        matDfaceH_idx                     = np.arange( (self.N+2)*(self.M+3), dtype=np.int32 )   ## Create full grid index for vertical faces
        matDfaceH_idx                     = matDfaceH_idx.reshape( (self.N+2,self.M+3) )         ## Reshape it into grid
        matDfaceH_idx[:,0] = matDfaceH_idx[:,-1]= -1
        matDfacemaskH_idx = matDfaceH_idx >= 0
        matDfaceH_idx[matDfacemaskH_idx] -= 1
        for i in range(1,self.N+2):
            matDfaceH_idx[i][matDfacemaskH_idx[i]] -= 2*i

        matDfaceV_idx                     = np.arange( (self.N+3)*(self.M+2), dtype=np.int32 )   ## Create full grid index for vertical faces
        matDfaceV_idx                     = matDfaceV_idx.reshape( (self.N+3,self.M+2) )         ## Reshape it into grid
        matDfaceV_idx[0,:] = matDfaceV_idx[-1,:] = -1
        matDfacemaskV_idx = matDfaceV_idx >= 0
        matDfaceV_idx[matDfacemaskV_idx] -= self.M+1
        matDfaceV_idx[matDfacemaskV_idx] += matDfaceH_idx.max()

        matDface_idx[dir2D.H] = matDfaceH_idx
        matDface_idx[dir2D.V] = matDfaceV_idx

        # print(matDfaceH_idx)
        # print(matDfaceV_idx)
        # print(matDvert_idx)
        #### =============================== ####
        #### dual cell-level data generation ####
        #### =============================== ####
        ## The cells of the dual grid coincide with the **numbering** of the primal vertices.
        matDcell_idx = np.copy(matPvert_idx)

        #### ================================= ####
        #### dual connectivity data generation ####
        #### ================================= ####
        ## Generate dual cell-face connectivity data
        matDcellface_idx               = np.empty( (4,(self.N+1),(self.M+1)), dtype=np.int32 )
        matDcellface_idx[dir2D.N,:,:]  = matDface_idx[dir2D.H][1:,1:-1];
        matDcellface_idx[dir2D.E,:,:]  = matDface_idx[dir2D.V][1:-1,1:];
        matDcellface_idx[dir2D.S,:,:]  = matDface_idx[dir2D.H][:-1,1:-1];
        matDcellface_idx[dir2D.W,:,:]  = matDface_idx[dir2D.V][1:-1,:-1];

        ## Generate dual vertex-face connectivity data
        matDvertface_idx               = np.empty( (4,(self.N+2),(self.M+2)), dtype=np.int32 )
        matDvertface_idx[dir2D.N,:,:]  = matDface_idx[dir2D.V][1: ,:];
        matDvertface_idx[dir2D.E,:,:]  = matDface_idx[dir2D.H][:  ,1:];
        matDvertface_idx[dir2D.S,:,:]  = matDface_idx[dir2D.V][:-1,:];
        matDvertface_idx[dir2D.W,:,:]  = matDface_idx[dir2D.H][:  ,:-1];

        ## Generate dual cell-vertex connectivity data
        matDcellvert_idx               = np.empty( (4,(self.N+1),(self.M+1)), dtype=np.int32 )
        matDcellvert_idx[dir2D.SW,:,:] = matDvert_idx[:-1,:-1];
        matDcellvert_idx[dir2D.SE,:,:] = matDvert_idx[1: ,:-1];
        matDcellvert_idx[dir2D.NE,:,:] = matDvert_idx[1: , 1:];
        matDcellvert_idx[dir2D.NW,:,:] = matDvert_idx[1: ,:-1];

        #### =============================== ####
        #### primal boundary data generation ####
        #### =============================== ####
        ## Generate primal boundary cell data, simple slicing
        bPcells_idx = [tmp, tmp, tmp, tmp]
        bPcells_idx[dir2D.N] = matPcell_idx[-1,:][matPcellmask_idx[-1,:]]
        bPcells_idx[dir2D.E] = matPcell_idx[:,-1][matPcellmask_idx[:,-1]]
        bPcells_idx[dir2D.S] = matPcell_idx[0,:][matPcellmask_idx[0,:]]
        bPcells_idx[dir2D.W] = matPcell_idx[:,0][matPcellmask_idx[:,0]]

        ## Generate primal boundary face data, simple slicing
        bPface_idx = [tmp, tmp, tmp, tmp]
        bPface_idx[dir2D.N] = matPface_idx[dir2D.H][-1,:][matPfacemaskH_idx[-1,:]]
        bPface_idx[dir2D.E] = matPface_idx[dir2D.V][:,-1][matPfacemaskV_idx[:,-1]]
        bPface_idx[dir2D.S] = matPface_idx[dir2D.H][0,:][matPfacemaskH_idx[0,:]]
        bPface_idx[dir2D.W] = matPface_idx[dir2D.V][:,0][matPfacemaskV_idx[:,0]]

        #### ============================= ####
        #### dual boundary data generation ####
        #### ============================= ####
        ## Generate dual boundary face data, simple slicing
        bDface_idx = [tmp, tmp, tmp, tmp]
        bDface_idx[dir2D.N] = matDface_idx[dir2D.H][-1,:][matDfacemaskH_idx[-1,:]]
        bDface_idx[dir2D.E] = matDface_idx[dir2D.V][:,-1][matDfacemaskV_idx[:,-1]]
        bDface_idx[dir2D.S] = matDface_idx[dir2D.H][0,:][matDfacemaskH_idx[0,:]]
        bDface_idx[dir2D.W] = matDface_idx[dir2D.V][:,0][matDfacemaskV_idx[:,0]]

        #### ============================ ####
        #### finalize post-initialization ####
        #### ============================ ####
        self.bPcells_idx        = bPcells_idx
        self.bPface_idx         = bPface_idx
        self.bDface_idx         = bDface_idx

        self.matPcell_idx       = matPcell_idx
        self.matPface_idx       = matPface_idx
        self.matPvert_idx       = matPvert_idx
        self.matPcellface_idx   = matPcellface_idx
        self.matPvertface_idx   = matPvertface_idx
        self.matPcellvert_idx   = matPcellvert_idx

        self.matDcell_idx       = matDcell_idx
        self.matDface_idx       = matDface_idx
        self.matDvert_idx       = matDvert_idx
        self.matDcellface_idx   = matDcellface_idx
        self.matDvertface_idx   = matDvertface_idx
        self.matDcellvert_idx   = matDcellvert_idx

    def __str__(self) -> str:
        return  f'===========================\n' + \
                f'2D Structured Grid (NxM):\nN = {self.N}\nM = {self.M}\n' + \
                f'===========================\n'

    def get_internal_primal_cells_idx(self) -> npt.NDArray[np.int32]:
        Pcells: npt.NDArray[np.int32] = self.matPcell_idx[1:-1,1:-1]
        return Pcells

    def get_boundary_primal_cells_idx(self) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32]]:

        flattened_idx: npt.NDArray[np.int32] = self.bPcells_idx[0][:]
        for idx in range(1,len(self.bPcells_idx)):
            flattened_idx = np.hstack( (flattened_idx, self.bPcells_idx[idx][:]) )

        return self.bPcells_idx, np.sort(flattened_idx)

    def get_boundary_primal_faces_idx(self) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32]]:

        flattened_idx: npt.NDArray[np.int32] = self.bPface_idx[0][:]
        for idx in range(1,len(self.bPface_idx)):
            flattened_idx = np.hstack( (flattened_idx, self.bPface_idx[idx][:]) )

        return self.bPface_idx, np.sort(flattened_idx)

    def get_all_primal_cells_idx(self) -> npt.NDArray[np.int32]:
        return self.matPcell_idx

    def get_all_primal_faces_idx(self) -> list[npt.NDArray[np.int32]]:
        return self.matPface_idx

    def get_all_primal_verts_idx(self) -> npt.NDArray[np.int32]:
        return self.matPvert_idx

    def get_all_primal_cell2face_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matPcellface_idx

    def get_all_primal_vert2face_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matPvertface_idx

    def get_internal_primal_cell2vert_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matPcellvert_idx

    def get_internal_primal_cell2face_connectivity(self) -> npt.NDArray[np.int32]:
        Pconnectivity : npt.NDArray[np.int32] = self.matPcellface_idx[:, 1:-1, 1:-1]
        return Pconnectivity

    def get_boundary_dual_faces_idx(self) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32]]:

        flattened_idx: npt.NDArray[np.int32] = self.bDface_idx[0][:]
        for idx in range(1,len(self.bDface_idx)):
            flattened_idx = np.hstack( (flattened_idx, self.bDface_idx[idx][:]) )

        return self.bDface_idx, np.sort(flattened_idx)

    def get_all_dual_cells_idx(self) -> npt.NDArray[np.int32]:
        return self.matDcell_idx

    # TODO update documentation
    def get_inner_dual_cells_idx(self) -> npt.NDArray[np.int32]:
        flattened_idx: npt.NDArray[np.int32] = self.matDcell_idx[0,:]
        flattened_idx = np.hstack((flattened_idx, self.matDcell_idx[-1,:]))
        flattened_idx = np.hstack((flattened_idx, self.matDcell_idx[:,0]))
        flattened_idx = np.hstack((flattened_idx, self.matDcell_idx[:,-1]))
        flattened_idx = np.sort(np.unique(flattened_idx))

        return flattened_idx

    def get_all_dual_faces_idx(self) -> list[npt.NDArray[np.int32]]:
        return self.matDface_idx

    def get_all_dual_verts_idx(self) -> npt.NDArray[np.int32]:
        return self.matDvert_idx

    def get_all_dual_cell2face_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matDcellface_idx

    def get_all_dual_vert2face_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matDvertface_idx

    def get_internal_dual_cell2vert_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matDcellvert_idx

if __name__ == "__main__":
    N = 3
    M = 3
    grid = Grid2D(N,M)
    #print(grid.get_boundary_primal_faces_idx())
