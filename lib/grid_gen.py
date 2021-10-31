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
    """Generates a two dimensional grid provided the number of interior cells in
       the x-y direction.

       Parameters
       ----------
       N : int
            Number of cells in the x-direction.
       M : int
            Number of cells in the y-direction.

       Class Members
       -------------
       get_internal_cells_idx() -> array_like ( [N, M] )
           returns indices of all interior cells in grid form.

       get_virtual_cells_idx() -> list[ array_like ] ( [4 -> [x]] )
           returns indices of all virtual cells in N,E,S,W format in grid form.

       get_virtual_faces_idx() -> list[ array_like ] ( [4 -> [x]] )
           returns indices of all virtual faces in N,E,S,W format in grid form.

       get_all_cells_idx() -> array_like ( [N+2, M+2] )
           returns indices of all cells in grid form.

       get_all_faces_idx() -> list[ array_like ] ( [2 -> [x,x]] )
           returns indices of all faces in V, H format in grid form.

       get_all_cell2face_connectivity() -> array_like ( [4, N+2, M+2] )
           returns indices of all faces connected to each cell in grid form.
           The first index refers to the direction (N E S W), and the other two
           indices refer to the cell index in the x-y frame.

       get_internal_cell2face_connectivity() -> array_like ( [4, N, M] )
           returns indices of all faces connected to each internal cell in grid
           form. The first index refers to the direction (N E S W), and the
           other two indices refer to the cell index in the x-y frame.

       Notes
       -----
       An index of -1 refers to a spurious datapoint (e.g. face or cell).

       Examples
       --------
       grid = Grid2D(64,32)
       internal_connectivity = grid.get_internal_cell2face_connectivity()
       """

    #### ============ ####
    #### Class inputs ####
    #### ============ ####
    N: int  ## Number of cells in the x-direction
    M: int  ## Number of cells in the y-direction

    #### =================== ####
    #### Post-init variables ####
    #### =================== ####
    matcell_idx:        npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [N+2, M+2]      // cell numbering, real, virtual, and spurious cells
    matface_idx:        list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [2 -> [x,x]]    // face numbering, real, virtual, and spurious faces
    matcellface_idx:    npt.NDArray[np.int32]       = field(init=False, repr=False)   ## [4, N+2, M+2]   // cell-face connectivity, real, virtual, and spurious faces
    vcells_idx:         list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // virtual cell indicess
    vface_idx:          list[npt.NDArray[np.int32]] = field(init=False, repr=False)   ## [4 -> [x]]      // virtual cell indicess

    def __post_init__(self) -> None:
        #### ======================== ####
        #### mypy variable annotation ####
        #### ======================== ####
        matcell_idx:        npt.NDArray[np.int32]
        matface_idx:        list[npt.NDArray[np.int32]]
        vcells_idx:         list[npt.NDArray[np.int32]]
        vface_idx:          list[npt.NDArray[np.int32]]
        matfaceH_idx:       npt.NDArray[np.int32]
        matfaceV_idx:       npt.NDArray[np.int32]
        matcellmask_idx:    npt.NDArray[np.bool_]
        matfacemaskH_idx:   npt.NDArray[np.bool_]
        matfacemaskV_idx:   npt.NDArray[np.bool_]
        tmp:                npt.NDArray[np.int32] = np.asarray(0, dtype=np.int32)

        #### ========================== ####
        #### cell-level data generation ####
        #### ========================== ####
        matcell_idx      = np.arange( (self.N+2)*(self.M+2), dtype=np.int32 )               ## Create full grid index (real, virtual, spurious cells)
        matcell_idx      = matcell_idx.reshape( (self.N+2,self.M+2) )                       ## Reshape it into a grid
        matcell_idx[0,0] = matcell_idx[-1,0] = matcell_idx[0,-1] = matcell_idx[-1,-1] = -1  ## Set spurious index to -1 -- always the corner cases
        matcellmask_idx  = matcell_idx >= 0                                                 ## Mask spurious cells

        matcell_idx[matcellmask_idx]           -= 1                                         ## Fix grid cell numbering such that index is continuous after disregarding spurious cells
        matcell_idx[1:][matcellmask_idx[1:]]   -= 1
        matcell_idx[-1:][matcellmask_idx[-1:]] -= 1

        #### ========================== ####
        #### face-level data generation ####
        #### ========================== ####
        matfaceH_idx                    = np.arange( (self.N+2)*(self.M+3), dtype=np.int32 )  ## Create full grid index for vertical faces
        matfaceH_idx                    = matfaceH_idx.reshape( (self.N+2,self.M+3) )         ## Reshape it into grid
        matfaceH_idx[0,:]               = matfaceH_idx[-1,:] = -1                             ## Set spurious index to -1
        matfacemaskH_idx                = matfaceH_idx >= 0                                   ## Mask spurious cells
        matfaceH_idx[matfacemaskH_idx] -= self.M+3

        matfaceV_idx                    = np.arange( (self.N+3)*(self.M+2), dtype=np.int32 )  ## Create full grid index for vertical faces
        matfaceV_idx                    = matfaceV_idx.reshape( (self.N+3,self.M+2) )         ## Reshape it into grid
        matfaceV_idx[:,0]               = matfaceV_idx[:,-1] = -1                             ## Set spurious index to -1
        matfacemaskV_idx                = matfaceV_idx >= 0                                   ## Mask spurious cells
        matfaceV_idx[matfacemaskV_idx] += matfaceH_idx.max()
        for i in range(1,self.M):
            matfaceV_idx[i][matfacemaskV_idx[i]] -= 2*i

        matface_idx          = [tmp, tmp]
        matface_idx[dir2D.H] = matfaceV_idx
        matface_idx[dir2D.V] = matfaceH_idx

        #### ============================ ####
        #### vertex-level data generation ####
        #### ============================ ####
        # TODO: implement vertex-level data generation, and then implement member functions to return the data

        #### ============================ ####
        #### connectivity data generation ####
        #### ============================ ####
        matcellface_idx              = np.empty( (4,(self.N+2),(self.M+2)), dtype = np.int32 )
        matcellface_idx[dir2D.N,:,:] = matfaceV_idx[1:,:];
        matcellface_idx[dir2D.E,:,:] = matfaceH_idx[:,1:];
        matcellface_idx[dir2D.S,:,:] = matfaceV_idx[:-1,:];
        matcellface_idx[dir2D.W,:,:] = matfaceH_idx[:,:-1];

        #### ======================= ####
        #### virtual data generation ####
        #### ======================= ####
        ## Generate virtual cell data, simple slicing
        vcells_idx = [tmp, tmp, tmp, tmp]
        vcells_idx[dir2D.N] = matcell_idx[-1,:][matcellmask_idx[-1,:]]
        vcells_idx[dir2D.E] = matcell_idx[:,-1][matcellmask_idx[:,-1]]
        vcells_idx[dir2D.S] = matcell_idx[0,:][matcellmask_idx[0,:]]
        vcells_idx[dir2D.W] = matcell_idx[:,0][matcellmask_idx[:,0]]

        ## Generate virtual face data, simple slicing
        vface_idx = [tmp, tmp, tmp, tmp]
        vface_idx[dir2D.N] = matfaceV_idx[-1,:][matfacemaskV_idx[-1,:]]
        vface_idx[dir2D.E] = matfaceH_idx[:,-1][matfacemaskH_idx[:,-1]]
        vface_idx[dir2D.S] = matfaceV_idx[0,:][matfacemaskV_idx[0,:]]
        vface_idx[dir2D.W] = matfaceH_idx[:,0][matfacemaskH_idx[:,0]]

        #### ============================ ####
        #### finalize post-initialization ####
        #### ============================ ####
        self.vcells_idx          = vcells_idx
        self.vface_idx           = vface_idx
        self.matcell_idx         = matcell_idx
        self.matface_idx         = matface_idx
        self.matcellface_idx     = matcellface_idx

    def __str__(self) -> str:
        return  f'===========================\n' + \
                f'2D Structured Grid (NxM):\nN = {self.N}\nM = {self.M}\n' + \
                f'===========================\n'

    def get_internal_cells_idx(self) -> npt.NDArray[np.int32]:
        cells : npt.NDArray[np.int32] = self.matcell_idx[1:-1,1:-1]
        return cells

    def get_virtual_cells_idx(self) -> list[npt.NDArray[np.int32]]:
        return self.vcells_idx

    def get_virtual_faces_idx(self) -> list[npt.NDArray[np.int32]]:
        return self.vface_idx

    def get_all_cells_idx(self) -> npt.NDArray[np.int32]:
        return self.matcell_idx

    def get_all_faces_idx(self) -> list[npt.NDArray[np.int32]]:
        return self.matface_idx

    def get_all_cell2face_connectivity(self) -> npt.NDArray[np.int32]:
        return self.matcellface_idx

    def get_internal_cell2face_connectivity(self) -> npt.NDArray[np.int32]:
        connectivity : npt.NDArray[np.int32] = self.matcellface_idx[:, 1:-1, 1:-1]
        return connectivity

if __name__ == "__main__":
    N = 3
    M = 4
    grid = Grid2D(N,M)

    print(type( grid.get_internal_cell2face_connectivity() ) )
