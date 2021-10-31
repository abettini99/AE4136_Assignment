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

       get_virtual_primal_cells_idx() -> list[ array_like ] ( [4 -> [x]] )
           returns indices of all virtual primal cells in N,E,S,W format in grid
           form.

       get_virtual_primal_faces_idx() -> list[ array_like ] ( [4 -> [x]] )
           returns indices of all virtual primal faces in N,E,S,W format in grid
           form.

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
    # v     - virtual
    # mat   - matrix

    matPcell_idx:        npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+2, M+2]      // cell numbering, real, virtual, and spurious cells
    matPface_idx:        list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [2 -> [x,x]]    // face numbering, real, virtual, and spurious faces
    matPvert_idx:        npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+1, M+1]      // vertices numbering, real vertices
    matPcellface_idx:    npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+2, M+2]   // cell-face connectivity, real, virtual, and spurious faces
    matPvertface_idx:    npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+1, M+1]   // vertex-face connectivity, real and spurious faces
    matPcellvert_idx:    npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N, M]       // cell-vertex connectivity, real cells
    vPcells_idx:         list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // virtual cell indicess
    vPface_idx:          list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // virtual cell indicess

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
        vPcells_idx:        list[npt.NDArray[np.int32]]
        vPface_idx:         list[npt.NDArray[np.int32]]
        matPfaceH_idx:      npt.NDArray[np.int32]
        matPfaceV_idx:      npt.NDArray[np.int32]
        matPcellmask_idx:   npt.NDArray[np.bool_]
        matPfacemaskH_idx:  npt.NDArray[np.bool_]
        matPfacemaskV_idx:  npt.NDArray[np.bool_]
        tmp:                npt.NDArray[np.int32] = np.asarray(0, dtype=np.int32)

        #### ================================= ####
        #### primal cell-level data generation ####
        #### ================================= ####
        matPcell_idx      = np.arange( (self.N+2)*(self.M+2), dtype=np.int32 )               ## Create full grid index (real, virtual, spurious cells)
        matPcell_idx      = matPcell_idx.reshape( (self.N+2,self.M+2) )                       ## Reshape it into a grid
        matPcell_idx[0,0] = matPcell_idx[-1,0] = matPcell_idx[0,-1] = matPcell_idx[-1,-1] = -1  ## Set spurious index to -1 -- always the corner cases
        matPcellmask_idx  = matPcell_idx >= 0                                                 ## Mask spurious cells

        matPcell_idx[matPcellmask_idx]           -= 1                                         ## Fix grid cell numbering such that index is continuous after disregarding spurious cells
        matPcell_idx[1:][matPcellmask_idx[1:]]   -= 1
        matPcell_idx[-1:][matPcellmask_idx[-1:]] -= 1

        #### ================================= ####
        #### primal face-level data generation ####
        #### ================================= ####
        matPfaceV_idx                     = np.arange( (self.N+2)*(self.M+3), dtype=np.int32 )  ## Create full grid index for vertical faces
        matPfaceV_idx                     = matPfaceV_idx.reshape( (self.N+2,self.M+3) )         ## Reshape it into grid
        matPfaceV_idx[0,:]                = matPfaceV_idx[-1,:] = -1                             ## Set spurious index to -1
        matPfacemaskV_idx                 = matPfaceV_idx >= 0                                   ## Mask spurious cells
        matPfaceV_idx[matPfacemaskV_idx] -= self.M+3

        matPfaceH_idx                     = np.arange( (self.N+3)*(self.M+2), dtype=np.int32 )  ## Create full grid index for vertical faces
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
        # TODO: Most likely wrong at the moment, probably requires virtual cells as well, even in primal form
        matPvert_idx                    = np.arange( (self.N+1)*(self.M+1), dtype=np.int32 )  ## Create full grid index for vertices
        matPvert_idx                    = matPvert_idx.reshape( (self.N+1,self.M+1) )          ## Reshape it into grid

        #### ============================ ####
        #### connectivity data generation ####
        #### ============================ ####
        ## Generate primal cell-face connectivity data
        matPcellface_idx               = np.empty( (4,(self.N+2),(self.M+2)), dtype = np.int32 )
        matPcellface_idx[dir2D.N,:,:]  = matPface_idx[dir2D.H][1:,:];
        matPcellface_idx[dir2D.E,:,:]  = matPface_idx[dir2D.V][:,1:];
        matPcellface_idx[dir2D.S,:,:]  = matPface_idx[dir2D.H][:-1,:];
        matPcellface_idx[dir2D.W,:,:]  = matPface_idx[dir2D.V][:,:-1];

        ## Generate primal vertex-face connectivity data
        matPvertface_idx               = np.empty( (4,(self.N+1),(self.M+1)), dtype = np.int32 )
        matPvertface_idx[dir2D.N,:,:]  = matPface_idx[dir2D.V][1:,1:-1];
        matPvertface_idx[dir2D.E,:,:]  = matPface_idx[dir2D.H][1:-1,1:];
        matPvertface_idx[dir2D.S,:,:]  = matPface_idx[dir2D.V][:-1,1:-1];
        matPvertface_idx[dir2D.W,:,:]  = matPface_idx[dir2D.H][1:-1,:-1];

        ## Generate primal cell-vertex connectivity data
        matPcellvert_idx               = np.empty( (4,(self.N),(self.M)), dtype = np.int32 )
        matPcellvert_idx[dir2D.SW,:,:] = matPvert_idx[:-1,:-1];
        matPcellvert_idx[dir2D.SE,:,:] = matPvert_idx[1: ,:-1];
        matPcellvert_idx[dir2D.NE,:,:] = matPvert_idx[1: , 1:];
        matPcellvert_idx[dir2D.NW,:,:] = matPvert_idx[1: ,:-1];

        #### ======================= ####
        #### virtual data generation ####
        #### ======================= ####
        ## Generate virtual primal cell data, simple slicing
        vPcells_idx = [tmp, tmp, tmp, tmp]
        vPcells_idx[dir2D.N] = matPcell_idx[-1,:][matPcellmask_idx[-1,:]]
        vPcells_idx[dir2D.E] = matPcell_idx[:,-1][matPcellmask_idx[:,-1]]
        vPcells_idx[dir2D.S] = matPcell_idx[0,:][matPcellmask_idx[0,:]]
        vPcells_idx[dir2D.W] = matPcell_idx[:,0][matPcellmask_idx[:,0]]

        ## Generate virtual face data, simple slicing
        vPface_idx = [tmp, tmp, tmp, tmp]
        vPface_idx[dir2D.N] = matPface_idx[dir2D.H][-1,:][matPfacemaskH_idx[-1,:]]
        vPface_idx[dir2D.E] = matPface_idx[dir2D.V][:,-1][matPfacemaskV_idx[:,-1]]
        vPface_idx[dir2D.S] = matPface_idx[dir2D.H][0,:][matPfacemaskH_idx[0,:]]
        vPface_idx[dir2D.W] = matPface_idx[dir2D.V][:,0][matPfacemaskV_idx[:,0]]

        #### ============================ ####
        #### finalize post-initialization ####
        #### ============================ ####
        self.vPcells_idx        = vPcells_idx
        self.vPface_idx         = vPface_idx
        self.matPcell_idx       = matPcell_idx
        self.matPface_idx       = matPface_idx
        self.matPvert_idx       = matPvert_idx
        self.matPcellface_idx   = matPcellface_idx
        self.matPvertface_idx   = matPvertface_idx
        self.matPcellvert_idx   = matPcellvert_idx

    def __str__(self) -> str:
        return  f'===========================\n' + \
                f'2D Structured Grid (NxM):\nN = {self.N}\nM = {self.M}\n' + \
                f'===========================\n'

    def get_internal_primal_cells_idx(self) -> npt.NDArray[np.int32]:
        cells : npt.NDArray[np.int32] = self.matPcell_idx[1:-1,1:-1]
        return cells

    def get_virtual_primal_cells_idx(self) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32]]:

        flattened_idx: npt.NDArray[np.int32] = self.vPcells_idx[0][:]
        for idx in range(1,len(self.vPcells_idx)):
            flattened_idx = np.hstack( (flattened_idx, self.vPcells_idx[idx][:]) )

        return self.vPcells_idx, np.sort(flattened_idx)

    def get_virtual_primal_faces_idx(self) -> tuple[list[npt.NDArray[np.int32]], npt.NDArray[np.int32]]:

        flattened_idx: npt.NDArray[np.int32] = self.vPface_idx[0][:]
        for idx in range(1,len(self.vPface_idx)):
            flattened_idx = np.hstack( (flattened_idx, self.vPface_idx[idx][:]) )

        return self.vPface_idx, np.sort(flattened_idx)

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

if __name__ == "__main__":
    N = 6
    M = 3
    grid = Grid2D(N,M)
    #print(grid.get_virtual_primal_faces_idx())
