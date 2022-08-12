#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from utils import *
from cell import *
from face import *
from vertex import *

# Functions / Classes
@dataclass() # Not super necessary to have this decorator for this to work, but it is still nice to have.
class Grid2D:

    #### ============ ####
    #### Class inputs ####
    #### ============ ####
    N: int  ## Number of cells in the x-direction
    M: int  ## Number of cells in the y-direction
    xp: npt.NDArray[float] = field(repr=False, default=np.asarray([]))
    xd: npt.NDArray[float] = field(repr=False, default=np.asarray([]))
    yp: npt.NDArray[float] = field(repr=False, default=np.asarray([]))
    yd: npt.NDArray[float] = field(repr=False, default=np.asarray([]))

    #### =================== ####
    #### Post-init variables ####
    #### =================== ####
    # Abbreviations
    # P     - primal
    # D     - dual
    # C     - cell
    # F     - face
    # V     - vertex
    Pcell_array: npt.NDArray[Cell] = field(init=False, repr=False)
    Pface_array: npt.NDArray[Face] = field(init=False, repr=False)
    Pvert_array: npt.NDArray[Vertex] = field(init=False, repr=False)
    Dcell_array: npt.NDArray[Cell] = field(init=False, repr=False)
    Dface_array: npt.NDArray[Face] = field(init=False, repr=False)
    Dvert_array: npt.NDArray[Vertex] = field(init=False, repr=False)

    def __post_init__(self) -> None:

        N = self.N
        M = self.M

        if self.xp.shape != (0,):
            if self.xp.shape != (M+1,):
                raise ValueError(f'xp length does not match number of real primal vertices in the x-direction (M+1)')

        if self.yp.shape != (0,):
            if self.yp.shape != (N+1,):
                raise ValueError(f'yp length does not match number of real primal vertices in the y-direction (N+1)')

        if self.xd.shape != (0,):
            if self.xd.shape != (M+2,):
                raise ValueError(f'xd length does not match number of real dual vertices in the x-direction (M+2)')

        if self.yd.shape != (0,):
            if self.yd.shape != (N+2,):
                raise ValueError(f'yd length does not match number of real dual vertices in the y-direction (N+2)')

        #### ============================================================== ####
        ####                    Create primal domain                        ####
        #### ============================================================== ####

        PCgrid = np.empty((N+4,M+4),dtype=object)       # +2 for virtual cells, +2 extra for ghost cells (simplifies construction)
        PFgrid_EW = np.empty((N+4,M+5),dtype=object)    # +1 always compared to #virtual+real cells, +2 for virtual cells, +2 for ghost cells (not necessary, but simplifies)
        PFgrid_NS = np.empty((N+5,M+4),dtype=object)
        PVgrid  = np.empty((N+5,M+5),dtype=object)      # +1 always compared to #virtual+real cells, +2 for virtual cells, +2 for ghost cells (not necessary, but simplifies)

        #### ============ ####
        #### Create cells ####
        #### ============ ####
        idx = 0
        for i in range(N+4):
            for j in range(M+4):
                ## Setup ghost cells
                if (i == 0 or i == N+3) or (j == 0 or j == M+3):
                    PCgrid[i,j] = Cell(2,'ghost')
                elif (i == 1 or i == N+2) and (j == 1 or j == M+2):
                    PCgrid[i,j] = Cell(2,'ghost')

                ## Setup virtual cells
                elif (i == 1 or i == N+2) or (j == 1 or j == M+2):
                    PCgrid[i,j] = Cell(2,'virtual')
                    PCgrid[i,j].set_idx(idx)
                    idx += 1
                ## Setup real cells
                else:
                    PCgrid[i,j] = Cell(2,'internal')
                    PCgrid[i,j].set_idx(idx)
                    idx += 1

        #### ============ ####
        #### Create faces ####
        #### ============ ####
        ## Assume all faces are first ghosts, we need to establish cells first
        for i in range(N+4):
            for j in range(M+5):
                PFgrid_EW[i,j] = Face(2,'ghost')

        for i in range(N+5):
            for j in range(M+4):
                PFgrid_NS[i,j] = Face(2,'ghost')

        #### =============== ####
        #### Create vertices ####
        #### =============== ####
        ## Assume all vertices are first ghosts, we need to establish cells and faces first
        for i in range(N+5):
            for j in range(M+5):
                PVgrid[i,j] = Vertex(2,'ghost')

        #### ========================= ####
        #### Establish cell neighbours ####
        #### ========================= ####
        for i in range(1,N+3):
            for j in range(1,M+3):
                ## Disregard all ghost cells
                if not PCgrid[i,j].type == 'ghost':
                    ## Real cell neighbour setup
                    if PCgrid[i,j].type == 'internal':
                        PCgrid[i,j].neighbours_idx[dir2D.N] = PCgrid[i+1,j].idx
                        PCgrid[i,j].neighbours_idx[dir2D.E] = PCgrid[i,j+1].idx
                        PCgrid[i,j].neighbours_idx[dir2D.S] = PCgrid[i-1,j].idx
                        PCgrid[i,j].neighbours_idx[dir2D.W] = PCgrid[i,j-1].idx
                    ## Virtual cell neighbour setup, we disregard neighbouring virtual cells as actual true neighbours
                    elif PCgrid[i,j].type == 'virtual':
                        if (PCgrid[i+1,j].type != 'ghost' and PCgrid[i+1,j].type != 'virtual'): PCgrid[i,j].neighbours_idx[dir2D.N] = PCgrid[i+1,j].idx
                        if (PCgrid[i,j+1].type != 'ghost' and PCgrid[i,j+1].type != 'virtual'): PCgrid[i,j].neighbours_idx[dir2D.E] = PCgrid[i,j+1].idx
                        if (PCgrid[i-1,j].type != 'ghost' and PCgrid[i-1,j].type != 'virtual'): PCgrid[i,j].neighbours_idx[dir2D.S] = PCgrid[i-1,j].idx
                        if (PCgrid[i,j-1].type != 'ghost' and PCgrid[i,j-1].type != 'virtual'): PCgrid[i,j].neighbours_idx[dir2D.W] = PCgrid[i,j-1].idx
                else:
                    pass

        # grid_tmp = np.empty((N+4,M+4,5))
        # for i in range(N+4):
        #     for j in range(M+4):
        #         grid_tmp[i,j,0] = PCgrid[i,j].sort_index
        #         grid_tmp[i,j,1] = PCgrid[i,j].neighbours_idx[dir2D.N]
        #         grid_tmp[i,j,2] = PCgrid[i,j].neighbours_idx[dir2D.E]
        #         grid_tmp[i,j,3] = PCgrid[i,j].neighbours_idx[dir2D.S]
        #         grid_tmp[i,j,4] = PCgrid[i,j].neighbours_idx[dir2D.W]
        # print(grid_tmp[:,:,0][1:-1,1:-1])
        # print(grid_tmp[:,:,1][1:-1,1:-1])
        # print(grid_tmp[:,:,2][1:-1,1:-1])
        # print(grid_tmp[:,:,3][1:-1,1:-1])
        # print(grid_tmp[:,:,4][1:-1,1:-1])

        #### ==================== ####
        #### Establish cell faces ####
        #### ==================== ####
        ## Note that we always establish the east/north face of the virtual/real cell in question!
        idx = 0
        ## All east-west cells first.
        for i in range(1,N+3):
            for j in range(1,M+3):
                ## Disregard all ghost cells
                if not PCgrid[i,j].type == 'ghost':
                    ## We have to establish all edge cases (first cell, interior cells, last cell)
                    ## First cell setup
                    if PCgrid[i,j].type == 'virtual' and PCgrid[i,j+1].type == 'internal':
                        ## Create face information
                        PFgrid_EW[i,j]                      = Face(2,'west')
                        PFgrid_EW[i,j+1]                    = Face(2,'internal')
                        PFgrid_EW[i,j].set_idx(idx); idx    += 1
                        PFgrid_EW[i,j+1].set_idx(idx); idx  += 1

                        PFgrid_EW[i,j].cells_idx[dir2D.Rp]   = PCgrid[i,j].idx # west-boundary face neighbours ghost cell
                        PFgrid_EW[i,j+1].cells_idx[dir2D.Lp] = PCgrid[i,j].idx
                        PFgrid_EW[i,j+1].cells_idx[dir2D.Rp] = PCgrid[i,j].neighbours_idx[dir2D.E]

                        ## Propagate face index to cells
                        PCgrid[i,j].faces_idx[dir2D.W]      = PFgrid_EW[i,j].idx
                        PCgrid[i,j].faces_idx[dir2D.E]      = PFgrid_EW[i,j+1].idx

                    ## Internal cell setup
                    elif PCgrid[i,j].type == 'internal' and (PCgrid[i,j+1].type == 'internal' or PCgrid[i,j+1].type == 'virtual'):
                        ## Create face information
                        PFgrid_EW[i,j+1]                    = Face(2,'internal')
                        PFgrid_EW[i,j+1].set_idx(idx); idx  += 1
                        PFgrid_EW[i,j+1].cells_idx[dir2D.Lp] = PCgrid[i,j].idx
                        PFgrid_EW[i,j+1].cells_idx[dir2D.Rp] = PCgrid[i,j].neighbours_idx[dir2D.E]

                        ## Propagate face index to cells
                        PCgrid[i,j].faces_idx[dir2D.W]      = PFgrid_EW[i,j].idx
                        PCgrid[i,j].faces_idx[dir2D.E]      = PFgrid_EW[i,j+1].idx

                    ## Last cell setup
                    elif PCgrid[i,j].type == 'virtual' and PCgrid[i,j-1].type == 'internal':
                        ## Create face information
                        PFgrid_EW[i,j+1]                    = Face(2,'east')
                        PFgrid_EW[i,j+1].set_idx(idx); idx  += 1
                        PFgrid_EW[i,j+1].cells_idx[dir2D.Lp] = PCgrid[i,j].idx
                        PFgrid_EW[i,j+1].cells_idx[dir2D.Rp] = PCgrid[i,j].neighbours_idx[dir2D.E]

                        ## Propagate face index to cells
                        PCgrid[i,j].faces_idx[dir2D.W]      = PFgrid_EW[i,j].idx
                        PCgrid[i,j].faces_idx[dir2D.E]      = PFgrid_EW[i,j+1].idx
                else:
                    pass

        # grid_tmp = -np.ones((N+4,M+5,1))
        # for i in range(N+4):
        #     # for j in range(M+5):
        #     #     grid_tmp[i,j,0]   = PFgrid_EW[i,j].idx
        #     for j in range(M+4):
        #         if PCgrid[i,j].type != 'ghost':
        #             grid_tmp[i,j+1,0]   = PCgrid[i,j].faces_idx[dir2D.E]
        #             grid_tmp[i,j,0]     = PCgrid[i,j].faces_idx[dir2D.W]
        # print(grid_tmp[:,:,0])

        ## All north-South cells next.
        for i in range(1,N+3):
            for j in range(1,M+3):
                ## Disregard all ghost cells
                if not PCgrid[i,j].type == 'ghost':
                    ## We have to establish all edge cases (first cell, interior cells, last cell)
                    ## First cell setup
                    if PCgrid[i,j].type == 'virtual' and PCgrid[i+1,j].type == 'internal':
                        ## Create face information
                        PFgrid_NS[i,j]                      = Face(2,'south')
                        PFgrid_NS[i+1,j]                    = Face(2,'internal')
                        PFgrid_NS[i,j].set_idx(idx); idx    += 1            # To get the correct numbering on index, the numbering has to be corrected on the next pass over (so we only name the south-faces)

                        PFgrid_NS[i,j].cells_idx[dir2D.Tp]   = PCgrid[i,j].idx # south-boundary face neighbours ghost cell
                        PFgrid_NS[i+1,j].cells_idx[dir2D.Bp] = PCgrid[i,j].idx
                        PFgrid_NS[i+1,j].cells_idx[dir2D.Tp] = PCgrid[i,j].neighbours_idx[dir2D.N]

                        ## Propagate face index to cells
                        PCgrid[i,j].faces_idx[dir2D.S]      = PFgrid_NS[i,j].idx
                        # PCgrid[i,j].faces_idx[dir2D.N]      = PFgrid_NS[i+1,j].idx

                    ## Internal cell setup
                    elif PCgrid[i,j].type == 'internal' and (PCgrid[i+1,j].type == 'internal' or PCgrid[i+1,j].type == 'virtual'):
                        ## Create face information
                        PFgrid_NS[i+1,j]                    = Face(2,'internal')
                        PFgrid_NS[i,j].set_idx(idx); idx  += 1
                        PFgrid_NS[i+1,j].cells_idx[dir2D.Bp] = PCgrid[i,j].idx
                        PFgrid_NS[i+1,j].cells_idx[dir2D.Tp] = PCgrid[i,j].neighbours_idx[dir2D.N]

                        ## Propagate face index to cells
                        PCgrid[i,j].faces_idx[dir2D.S]      = PFgrid_NS[i,j].idx
                        PCgrid[i-1,j].faces_idx[dir2D.N]      = PFgrid_NS[i,j].idx

                    ## Last cell setup
                    elif PCgrid[i,j].type == 'virtual' and PCgrid[i-1,j].type == 'internal':
                        ## Create face information
                        PFgrid_NS[i+1,j]                    = Face(2,'north')
                        PFgrid_NS[i,j].set_idx(idx)
                        PFgrid_NS[i+1,j].set_idx(idx+M); idx  += 1 # Quick fix
                        PFgrid_NS[i+1,j].cells_idx[dir2D.Bp] = PCgrid[i,j].idx
                        PFgrid_NS[i+1,j].cells_idx[dir2D.Tp] = PCgrid[i,j].neighbours_idx[dir2D.N]

                        ## Propagate face index to cells
                        PCgrid[i,j].faces_idx[dir2D.S]      = PFgrid_NS[i,j].idx
                        PCgrid[i-1,j].faces_idx[dir2D.N]    = PFgrid_NS[i,j].idx
                        PCgrid[i,j].faces_idx[dir2D.N]      = PFgrid_NS[i+1,j].idx
                else:
                    pass

        # grid_tmp = -np.ones((N+5,M+4,1))
        # # for i in range(N+5):
        #     # for j in range(M+4):
        #         # grid_tmp[i,j,0]   = PFgrid_NS[i,j].idx
        # for i in range(N+4):
        #     for j in range(M+4):
        #         if PCgrid[i,j].type != 'ghost':
        #             grid_tmp[i+1,j,0]   = PCgrid[i,j].faces_idx[dir2D.N]
        #             grid_tmp[i,j,0]     = PCgrid[i,j].faces_idx[dir2D.S]
        # print(grid_tmp[:,:,0])

        #### ================== ####
        #### Establish vertices ####
        #### ================== ####
        ## Establishing vertices has too many laws for a good implementation currently. Done by brute-forcing with one or two laws.
        idx = 0
        ## First establish all vertex points associated to real vertex points, starting from bottom left to top right
        for i in range(1,N+3):
            for j in range(1,M+3):
                if PCgrid[i,j].type == 'internal':
                    PVgrid[i,j] = Vertex(2, 'internal')       # SW
                    PVgrid[i,j+1] = Vertex(2, 'internal')     # SE
                    PVgrid[i+1,j+1] = Vertex(2, 'internal')   # NE
                    PVgrid[i+1,j] = Vertex(2, 'internal')     # NW
                else:
                    pass
        ## Establish all vertex points associated with virtual cells by using real vertex points
        for i in range(1,N+3):
            for j in range(1,M+3):
                # Check if east, west vertices belong to an internal cell, then set the vertices to the south/north
                if PCgrid[i,j].type == 'internal':
                    if PCgrid[i-1,j].type == 'virtual': PVgrid[i-1,j] = Vertex(2, 'south'); PVgrid[i-1,j+1] = Vertex(2, 'south')
                    elif PCgrid[i+1,j].type == 'virtual': PVgrid[i+2,j] = Vertex(2, 'north'); PVgrid[i+2,j+1] = Vertex(2, 'north')

                    if PCgrid[i,j-1].type == 'virtual': PVgrid[i,j-1] = Vertex(2, 'west'); PVgrid[i+1,j-1] = Vertex(2, 'west')
                    elif PCgrid[i,j+1].type == 'virtual': PVgrid[i,j+2] = Vertex(2, 'east'); PVgrid[i+1,j+2] = Vertex(2, 'east')
                else:
                    pass

        ## Then establish all cell idx attached to vertex points
        for i in range(1,N+3):
            for j in range(1,M+3):
                if not PCgrid[i,j].type == 'ghost':
                    ## Establish all cell indices already
                    PVgrid[i,j].cells_idx[dir2D.NE] = PCgrid[i,j].idx   # relative to SW point, NE cell is cell idx
                    PVgrid[i,j+1].cells_idx[dir2D.NW] = PCgrid[i,j].idx  # relative to SE point, NW cell is cell idx
                    PVgrid[i+1,j+1].cells_idx[dir2D.SW] = PCgrid[i,j].idx  # relative to NE point, SW cell is cell idx
                    PVgrid[i+1,j].cells_idx[dir2D.SE] = PCgrid[i,j].idx  # relative to NW point, SE cell is cell idx
                else:
                    pass
        ## Add indices to vertices
        idx = 0
        for i in range(1,N+5):
            for j in range(1,M+5):
                if not PVgrid[i,j].type == 'ghost':
                    PVgrid[i,j].idx = idx; idx += 1
                else:
                    pass
        # grid_tmp = np.empty((N+5,M+5))
        # for i in range(N+5):
        #     for j in range(M+5):
        #         grid_tmp[i,j]   = PVgrid[i,j].idx
        # print(grid_tmp)

        ## Now establish all vertex idx attached to cell
        for i in range(1,N+3):
            for j in range(1,M+3):
                if not PCgrid[i,j].type == 'ghost':
                    ## Establish all cell indices already
                    PCgrid[i,j].vertices_idx[dir2D.SW] = PVgrid[i,j].idx
                    PCgrid[i,j].vertices_idx[dir2D.SE] = PVgrid[i,j+1].idx
                    PCgrid[i,j].vertices_idx[dir2D.NE] = PVgrid[i+1,j+1].idx
                    PCgrid[i,j].vertices_idx[dir2D.NW] = PVgrid[i+1,j].idx
                else:
                    pass
        ## Now establish all vertex points attached to faces, and vice-versa
        for i in range(1,N+4):
            for j in range(1,M+4):
                if not PFgrid_EW[i,j].type == 'ghost':
                    ## Establish vertices attached to faces
                    PFgrid_EW[i,j].vertices_idx[dir2D.Bp] = PVgrid[i,j].idx
                    PFgrid_EW[i,j].vertices_idx[dir2D.Tp] = PVgrid[i+1,j].idx
                    ## And establish faces attached to vertices
                    PVgrid[i,j].faces_idx[dir2D.N] = PFgrid_EW[i,j].idx
                    PVgrid[i,j].faces_idx[dir2D.S] = PFgrid_EW[i-1,j].idx
                if not PFgrid_NS[i,j].type == 'ghost':
                    ## Establish vertices attached to faces
                    PFgrid_NS[i,j].vertices_idx[dir2D.Lp] = PVgrid[i,j].idx
                    PFgrid_NS[i,j].vertices_idx[dir2D.Rp] = PVgrid[i,j+1].idx
                    ## And establish faces attached to vertices
                    PVgrid[i,j].faces_idx[dir2D.E] = PFgrid_NS[i,j].idx
                    PVgrid[i,j].faces_idx[dir2D.W] = PFgrid_NS[i,j-1].idx

        # grid_tmp = np.empty((N+5,M+4,1))
        # # for i in range(N+5):
        #     # for j in range(M+4):
        #         # grid_tmp[i,j,0]   = PFgrid_NS[i,j].idx
        # for i in range(N+4):
        #     for j in range(M+4):
        #         grid_tmp[i+1,j,0]   = PCgrid[i,j].faces_idx[dir2D.N]
        #         grid_tmp[i,j,0]     = PCgrid[i,j].faces_idx[dir2D.S]
        # print(grid_tmp[:,:,0])

        # grid_tmp = -1*np.ones((N+5,M+5,1))
        # for i in range(N+4):
        #     for j in range(M+4):
        #         if not PCgrid[i,j].type == 'ghost':
        #             grid_tmp[i,j,0]   = PCgrid[i,j].vertices_idx[dir2D.SW]
        #             grid_tmp[i,j+1,0]   = PCgrid[i,j].vertices_idx[dir2D.SE]
        #             grid_tmp[i+1,j+1,0]   = PCgrid[i,j].vertices_idx[dir2D.NE]
        #             grid_tmp[i+1,j,0]   = PCgrid[i,j].vertices_idx[dir2D.NW]
        # print(grid_tmp[:,:,0])

        #### ============================================================== ####
        ####                      Create dual domain                        ####
        #### ============================================================== ####

        DCgrid = np.empty((N+3,M+3),dtype=object)       # +1 always compared to primal grid, +2 extra for ghost cells (simplifies construction)
        DFgrid_EW = np.empty((N+3,M+4),dtype=object)    # +1 always compared to #virtual+real cells, +1 always compared to primal faces, +2 for ghost cells (not necessary, but simplifies)
        DFgrid_NS = np.empty((N+4,M+3),dtype=object)
        DVgrid  = np.empty((N+4,M+4),dtype=object)      # +1 always compared to #virtual+real cells, +1 always compared to primal faces, +2 for ghost cells (not necessary, but simplifies)

        #### ============ ####
        #### Create cells ####
        #### ============ ####
        idx = 0
        for i in range(N+3):
            for j in range(M+3):
                ## Setup ghost cells
                if (i == 0 or i == N+2) or (j == 0 or j == M+2):
                    DCgrid[i,j] = Cell(2,'ghost')

                ## Setup real cells
                else:
                    DCgrid[i,j] = Cell(2,'internal')
                    DCgrid[i,j].set_idx(idx)
                    idx += 1

        #### ============ ####
        #### Create faces ####
        #### ============ ####
        ## Assume all faces are first ghosts, we need to establish cells first
        for i in range(N+3):
            for j in range(M+4):
                DFgrid_EW[i,j] = Face(2,'ghost')

        for i in range(N+4):
            for j in range(M+3):
                DFgrid_NS[i,j] = Face(2,'ghost')

        #### =============== ####
        #### Create vertices ####
        #### =============== ####
        ## Assume all vertices are first ghosts, we need to establish cells and faces first
        for i in range(N+4):
            for j in range(M+4):
                DVgrid[i,j] = Vertex(2,'ghost')

        #### ========================= ####
        #### Establish cell neighbours ####
        #### ========================= ####
        for i in range(1,N+2):
            for j in range(1,M+2):
                ## Disregard all ghost cells
                if not DCgrid[i,j].type == 'ghost': # not needed, but keeps consistency with primal setup
                    ## Real cell neighbour setup
                    if DCgrid[i,j].type == 'internal':
                        DCgrid[i,j].neighbours_idx[dir2D.N] = DCgrid[i+1,j].idx
                        DCgrid[i,j].neighbours_idx[dir2D.E] = DCgrid[i,j+1].idx
                        DCgrid[i,j].neighbours_idx[dir2D.S] = DCgrid[i-1,j].idx
                        DCgrid[i,j].neighbours_idx[dir2D.W] = DCgrid[i,j-1].idx
                else:
                    pass

        #### ==================== ####
        #### Establish cell faces ####
        #### ==================== ####
        ## Note that we always establish the east/north face of the virtual/real cell in question!
        idx = 0
        ## All north-South cells first.
        for i in range(1,N+2):
            for j in range(1,M+2):
                ## Disregard all ghost cells
                if not DCgrid[i,j].type == 'ghost':
                    ## We have to establish all edge cases (first cell, interior cells, last cell)
                    ## First cell setup
                    if DCgrid[i-1,j].type == 'ghost' and DCgrid[i,j].type == 'internal':
                        ## Create face information
                        DFgrid_NS[i,j]                      = Face(2,'south')
                        DFgrid_NS[i+1,j]                    = Face(2,'internal')
                        DFgrid_NS[i,j].set_idx(idx); idx    += 1            # To get the correct numbering on index, the numbering has to be corrected on the next pass over (so we only name the south-faces)

                        DFgrid_NS[i,j].cells_idx[dir2D.Td]   = DCgrid[i,j].idx # south-boundary face neighbours ghost cell
                        DFgrid_NS[i+1,j].cells_idx[dir2D.Bd] = DCgrid[i,j].idx
                        DFgrid_NS[i+1,j].cells_idx[dir2D.Td] = DCgrid[i,j].neighbours_idx[dir2D.N]

                        ## Propagate face index to cells
                        DCgrid[i,j].faces_idx[dir2D.S]      = DFgrid_NS[i,j].idx
                        # DCgrid[i,j].faces_idx[dir2D.N]      = DFgrid_NS[i+1,j].idx

                    ## Internal cell setup
                    elif DCgrid[i,j].type == 'internal' and DCgrid[i+1,j].type == 'internal':
                        ## Create face information
                        DFgrid_NS[i+1,j]                    = Face(2,'internal')
                        DFgrid_NS[i,j].set_idx(idx); idx  += 1
                        DFgrid_NS[i+1,j].cells_idx[dir2D.Bd] = DCgrid[i,j].idx
                        DFgrid_NS[i+1,j].cells_idx[dir2D.Td] = DCgrid[i,j].neighbours_idx[dir2D.N]

                        ## Propagate face index to cells
                        DCgrid[i,j].faces_idx[dir2D.S]      = DFgrid_NS[i,j].idx
                        DCgrid[i-1,j].faces_idx[dir2D.N]    = DFgrid_NS[i,j].idx

                    ## Last cell setup
                    elif DCgrid[i,j].type == 'internal' and DCgrid[i+1,j].type == 'ghost':
                        ## Create face information
                        DFgrid_NS[i+1,j]                    = Face(2,'north')
                        DFgrid_NS[i,j].set_idx(idx)
                        DFgrid_NS[i+1,j].set_idx(idx+(M+1)); idx  += 1 # Quick fix
                        DFgrid_NS[i+1,j].cells_idx[dir2D.Bd] = DCgrid[i,j].idx
                        DFgrid_NS[i+1,j].cells_idx[dir2D.Td] = DCgrid[i,j].neighbours_idx[dir2D.N]

                        ## Propagate face index to cells
                        DCgrid[i,j].faces_idx[dir2D.S]      = DFgrid_NS[i,j].idx
                        DCgrid[i-1,j].faces_idx[dir2D.N]    = DFgrid_NS[i,j].idx
                        DCgrid[i,j].faces_idx[dir2D.N]      = DFgrid_NS[i+1,j].idx
                else:
                    pass

        ## All east-west cells next.
        idx += M+1
        for i in range(1,N+2):
            for j in range(1,M+2):
                ## Disregard all ghost cells
                if not DCgrid[i,j].type == 'ghost':
                    ## We have to establish all edge cases (first cell, interior cells, last cell)
                    ## First cell setup
                    if DCgrid[i,j-1].type == 'ghost' and DCgrid[i,j].type == 'internal':
                        ## Create face information
                        DFgrid_EW[i,j]                      = Face(2,'west')
                        DFgrid_EW[i,j+1]                    = Face(2,'internal')
                        DFgrid_EW[i,j].set_idx(idx); idx    += 1
                        DFgrid_EW[i,j+1].set_idx(idx); idx  += 1

                        DFgrid_EW[i,j].cells_idx[dir2D.Rd]   = DCgrid[i,j].idx # west-boundary face neighbours ghost cell
                        DFgrid_EW[i,j+1].cells_idx[dir2D.Ld] = DCgrid[i,j].idx
                        DFgrid_EW[i,j+1].cells_idx[dir2D.Rd] = DCgrid[i,j].neighbours_idx[dir2D.E]

                        ## Propagate face index to cells
                        DCgrid[i,j].faces_idx[dir2D.W]      = DFgrid_EW[i,j].idx
                        DCgrid[i,j].faces_idx[dir2D.E]      = DFgrid_EW[i,j+1].idx

                    ## Internal cell setup
                    elif DCgrid[i,j].type == 'internal' and DCgrid[i,j+1].type == 'internal':
                        ## Create face information
                        DFgrid_EW[i,j+1]                    = Face(2,'internal')
                        DFgrid_EW[i,j+1].set_idx(idx); idx  += 1
                        DFgrid_EW[i,j+1].cells_idx[dir2D.Ld] = DCgrid[i,j].idx
                        DFgrid_EW[i,j+1].cells_idx[dir2D.Rd] = DCgrid[i,j].neighbours_idx[dir2D.E]

                        ## Propagate face index to cells
                        DCgrid[i,j].faces_idx[dir2D.W]      = DFgrid_EW[i,j].idx
                        DCgrid[i,j].faces_idx[dir2D.E]      = DFgrid_EW[i,j+1].idx

                    ## Last cell setup
                    elif DCgrid[i,j].type == 'internal' and DCgrid[i,j+1].type == 'ghost':
                        ## Create face information
                        DFgrid_EW[i,j+1]                    = Face(2,'east')
                        DFgrid_EW[i,j+1].set_idx(idx); idx  += 1
                        DFgrid_EW[i,j+1].cells_idx[dir2D.Ld] = DCgrid[i,j].idx
                        DFgrid_EW[i,j+1].cells_idx[dir2D.Rd] = DCgrid[i,j].neighbours_idx[dir2D.E]

                        ## Propagate face index to cells
                        DCgrid[i,j].faces_idx[dir2D.W]      = DFgrid_EW[i,j].idx
                        DCgrid[i,j].faces_idx[dir2D.E]      = DFgrid_EW[i,j+1].idx
                else:
                    pass

        # grid_tmp = -np.ones((N+4,M+3))
        # for i in range(N+4):
        #     for j in range(M+3):
        #         grid_tmp[i,j]   = DFgrid_NS[i,j].idx
        # print(grid_tmp)
        # print()
        # grid_tmp = np.ones((N+3,M+4))
        # for i in range(N+3):
        #     for j in range(M+4):
        #         grid_tmp[i,j]   = DFgrid_EW[i,j].idx
        # print(grid_tmp)

        #### ================== ####
        #### Establish vertices ####
        #### ================== ####
        ## Establishing vertices has too many laws for a good implementation currently. Done by brute-forcing with one or two laws.
        idx = 0
        ## First establish all vertex points associated to real vertex points, starting from bottom left to top right
        for i in range(1,N+2):
            for j in range(1,M+2):
                if DCgrid[i,j].type == 'internal':
                    DVgrid[i,j] = Vertex(2, 'internal')       # SW
                    DVgrid[i,j+1] = Vertex(2, 'internal')     # SE
                    DVgrid[i+1,j+1] = Vertex(2, 'internal')   # NE
                    DVgrid[i+1,j] = Vertex(2, 'internal')     # NW
                else:
                    pass

        ## Establish all vertex points associated with ghost cells by using real vertex points
        for i in range(1,N+2):
            for j in range(1,M+2):
                # Check if east, west vertices belong to an internal cell, then set the vertices to the south/north
                if DCgrid[i,j].type == 'internal':
                    if DCgrid[i-1,j].type == 'ghost': DVgrid[i,j] = Vertex(2, 'south'); DVgrid[i,j+1] = Vertex(2, 'south')
                    elif DCgrid[i+1,j].type == 'ghost': DVgrid[i+1,j] = Vertex(2, 'north'); DVgrid[i+1,j+1] = Vertex(2, 'north')

                    if DCgrid[i,j-1].type == 'ghost': DVgrid[i,j] = Vertex(2, 'west'); DVgrid[i+1,j] = Vertex(2, 'west')
                    elif DCgrid[i,j+1].type == 'ghost': DVgrid[i,j+1] = Vertex(2, 'east'); DVgrid[i+1,j+1] = Vertex(2, 'east')
                else:
                    pass

        ## Then establish all cell idx attached to vertex points
        for i in range(1,N+2):
            for j in range(1,M+2):
                if not DCgrid[i,j].type == 'ghost':
                    ## Establish all cell indices already
                    DVgrid[i,j].cells_idx[dir2D.NE] = DCgrid[i,j].idx   # relative to SW point, NE cell is cell idx
                    DVgrid[i,j+1].cells_idx[dir2D.NW] = DCgrid[i,j].idx  # relative to SE point, NW cell is cell idx
                    DVgrid[i+1,j+1].cells_idx[dir2D.SW] = DCgrid[i,j].idx  # relative to NE point, SW cell is cell idx
                    DVgrid[i+1,j].cells_idx[dir2D.SE] = DCgrid[i,j].idx  # relative to NW point, SE cell is cell idx
                else:
                    pass

        ## Overwrite corner points to be ghost points:
            DVgrid[1,1].type = DVgrid[N+2,1].type = DVgrid[N+2,M+2].type = DVgrid[1,M+2].type = 'ghost'

        ## Add indices to vertices
        idx = 0
        for i in range(1,N+4):
            for j in range(1,M+4):
                if not DVgrid[i,j].type == 'ghost':
                    DVgrid[i,j].idx = idx; idx += 1
                else:
                    pass

        grid_tmp = -1*np.ones((N+4,M+4))
        for i in range(N+4):
            for j in range(M+4):
                if not DVgrid[i,j].type == 'ghost':
                    grid_tmp[i,j]   = DVgrid[i,j].idx
        print(grid_tmp)

        ## Now establish all vertex idx attached to cell
        for i in range(1,N+2):
            for j in range(1,M+2):
                if not DCgrid[i,j].type == 'ghost':
                    ## Establish all cell indices already
                    DCgrid[i,j].vertices_idx[dir2D.SW] = DVgrid[i,j].idx
                    DCgrid[i,j].vertices_idx[dir2D.SE] = DVgrid[i,j+1].idx
                    DCgrid[i,j].vertices_idx[dir2D.NE] = DVgrid[i+1,j+1].idx
                    DCgrid[i,j].vertices_idx[dir2D.NW] = DVgrid[i+1,j].idx
                else:
                    pass
        ## Now establish all vertex points attached to faces, and vice-versa
        for i in range(1,N+3):
            for j in range(1,M+3):

                ## Establish vertices attached to faces
                DFgrid_EW[i,j].vertices_idx[dir2D.Bd] = DVgrid[i,j].idx
                DFgrid_EW[i,j].vertices_idx[dir2D.Td] = DVgrid[i+1,j].idx
                ## And establish faces attached to vertices
                DVgrid[i,j].faces_idx[dir2D.N] = DFgrid_EW[i,j].idx
                DVgrid[i,j].faces_idx[dir2D.S] = DFgrid_EW[i-1,j].idx

                ## Establish vertices attached to faces
                DFgrid_NS[i,j].vertices_idx[dir2D.Ld] = DVgrid[i,j].idx
                DFgrid_NS[i,j].vertices_idx[dir2D.Rd] = DVgrid[i,j+1].idx
                ## And establish faces attached to vertices
                DVgrid[i,j].faces_idx[dir2D.E] = DFgrid_NS[i,j].idx
                DVgrid[i,j].faces_idx[dir2D.W] = DFgrid_NS[i,j-1].idx

        #
        # grid_tmp = -1*np.ones((N+4,M+4,1))
        # for i in range(N+3):
        #     for j in range(M+3):
        #         if not DCgrid[i,j].type == 'ghost':
        #             grid_tmp[i,j,0]   = DCgrid[i,j].vertices_idx[dir2D.SW]
        #             grid_tmp[i,j+1,0]   = DCgrid[i,j].vertices_idx[dir2D.SE]
        #             grid_tmp[i+1,j+1,0]   = DCgrid[i,j].vertices_idx[dir2D.NE]
        #             grid_tmp[i+1,j,0]   = DCgrid[i,j].vertices_idx[dir2D.NW]
        # print(grid_tmp[:,:,0])

        # grid_tmp = -1*np.ones((N+4,M+4))
        # for i in range(N+3):
        #     for j in range(N+4):
        #         if DFgrid_EW[i,j].type == 'internal':
        #             grid_tmp[i,j] = DFgrid_EW[i,j].vertices_idx[dir2D.Bd]
        #             grid_tmp[i+1,j] = DFgrid_EW[i,j].vertices_idx[dir2D.Td]
        # print(grid_tmp)
        #
        # grid_tmp = -1*np.ones((N+4,M+4))
        # for i in range(N+4):
        #     for j in range(N+3):
        #         if DFgrid_NS[i,j].type == 'internal':
        #             grid_tmp[i,j] = DFgrid_NS[i,j].vertices_idx[dir2D.Ld]
        #             grid_tmp[i,j+1] = DFgrid_NS[i,j].vertices_idx[dir2D.Rd]
        # print(grid_tmp)

        #### ============================================================== ####
        ####                        finalize domains                        ####
        #### ============================================================== ####

        ## Add coordinates to vertices
        # primal vertices
        if self.xp.shape != (0,) and self.yp.shape != (0,):
            xidx = 0; yidx = 0
            # Change all x
            for i in range(N+5):
                for j in range (M+5):
                    if PVgrid[i,j].type != 'ghost':
                        PVgrid[i,j].coordinates[dir2D.x] = self.xp[xidx]
                        if PVgrid[i,j].type != 'west' and PVgrid[i,j].type != 'east': xidx = min(xidx+1,len(self.xp)-1) # caps index to the length of xp
                xidx = 0
            # Change all y
            for j in range(M+5):
                for i in range (N+5):
                    if PVgrid[i,j].type != 'ghost':
                        PVgrid[i,j].coordinates[dir2D.y] = self.yp[yidx]
                        if PVgrid[i,j].type != 'north' and PVgrid[i,j].type != 'south': yidx = min(yidx+1,len(self.yp)-1)
                yidx = 0

            # grid_tmp = -1*np.ones((N+5,M+5),dtype=object)
            # for i in range(N+5):
            #     for j in range(M+5):
            #         grid_tmp[i,j]   = PVgrid[i,j].coordinates[dir2D.x]
            # print(grid_tmp)

        # dual vertices
        if self.xd.shape != (0,) and self.yd.shape != (0,):
            xidx = -1; yidx = -1 # HACKJOB: xidx and yidx are incremented regardless of ghost cell or not, so we start at -1 for this case so that when we reach real cells, we start at idx = 0
            # Change all x
            for i in range(N+4):
                for j in range (M+4):
                    if DVgrid[i,j].type != 'ghost':
                        DVgrid[i,j].coordinates[dir2D.x] = self.xd[xidx]
                    xidx = min(xidx+1,len(self.xd)-1)
                xidx = -1
            # Change all y
            for j in range(M+4):
                for i in range (N+4):
                    if DVgrid[i,j].type != 'ghost':
                        DVgrid[i,j].coordinates[dir2D.y] = self.yd[yidx]
                    yidx = min(yidx+1,len(self.yd)-1)
                yidx = -1

            # grid_tmp = -1*np.ones((N+4,M+4))
            # for i in range(N+4):
            #     for j in range(M+4):
            #         grid_tmp[i,j]   = DVgrid[i,j].idx
            # print(grid_tmp)

        ## Establish mirror index:
        for i in range(1,N+4):
            for j in range(1,M+5):
                if PFgrid_EW[i,j].type == 'internal':
                    PFgrid_EW[i,j].Midx = DFgrid_NS[i+1,j].idx
                    DFgrid_NS[i+1,j].Midx = PFgrid_EW[i,j].idx

                    print(PFgrid_EW[i,j].idx,PFgrid_EW[i,j].Midx)

        for i in range(1,N+5):
            for j in range(1,M+4):
                if PFgrid_NS[i,j].type == 'internal':
                    PFgrid_NS[i,j].Midx = DFgrid_EW[i,j+1].idx
                    DFgrid_EW[i,j+1].Midx = PFgrid_NS[i,j].idx



        self.Pcell_array = PCgrid.flatten()
        self.Pface_array = np.hstack((PFgrid_EW.flatten(),PFgrid_NS.flatten()))
        self.Pvert_array = PVgrid.flatten()
        self.Dcell_array = DCgrid.flatten()
        self.Dface_array = np.hstack((DFgrid_NS.flatten(),DFgrid_EW.flatten()))
        self.Dvert_array = DVgrid.flatten()

        ## Remove all ghost cells
        # Remove all ghost cells from Pcell_array
        tmp_mask = np.ones((len(self.Pcell_array)),dtype=bool)
        for idx, cell in enumerate(self.Pcell_array):
            if cell.type == 'ghost': tmp_mask[idx] = False
        self.Pcell_array = self.Pcell_array[tmp_mask]

        # Remove all ghost cells from Pface_array
        tmp_mask = np.ones((len(self.Pface_array)),dtype=bool)
        for idx, face in enumerate(self.Pface_array):
            if face.type == 'ghost': tmp_mask[idx] = False
        self.Pface_array = self.Pface_array[tmp_mask]

        # Remove all ghost cells from Pvert_array
        tmp_mask = np.ones((len(self.Pvert_array)),dtype=bool)
        for idx, vert in enumerate(self.Pvert_array):
            if vert.type == 'ghost': tmp_mask[idx] = False
        self.Pvert_array = self.Pvert_array[tmp_mask]

        # Remove all ghost cells from Dcell_array
        tmp_mask = np.ones((len(self.Dcell_array)),dtype=bool)
        for idx, cell in enumerate(self.Dcell_array):
            if cell.type == 'ghost': tmp_mask[idx] = False
        self.Dcell_array = self.Dcell_array[tmp_mask]

        # Remove all ghost cells from Dface_array
        tmp_mask = np.ones((len(self.Dface_array)),dtype=bool)
        for idx, face in enumerate(self.Dface_array):
            if face.type == 'ghost': tmp_mask[idx] = False
        self.Dface_array = self.Dface_array[tmp_mask]

        # Remove all ghost cells from Dvert_array
        tmp_mask = np.ones((len(self.Dvert_array)),dtype=bool)
        for idx, vert in enumerate(self.Dvert_array):
            if vert.type == 'ghost': tmp_mask[idx] = False
        self.Dvert_array = self.Dvert_array[tmp_mask]

        ## Add in truncated indices
        # Add all truncated indices to Pcell_array
        Iidx = 0; Bidx = 0
        for Pcell in self.Pcell_array:
            # if Pcell.type == 'internal':
            Pcell.set_Tidx(Iidx); Iidx += 1
            # else:
                # Pcell.set_Tidx(Bidx); Bidx += 1

        # Add all truncated indices to Pface_array
        Iidx = 0; Bidx = 0
        for Pface in self.Pface_array:
            if Pface.type == 'internal':
                Pface.set_Tidx(Iidx); Iidx += 1
            else:
                Pface.set_Tidx(Bidx); Bidx += 1

        # Add all truncated indices to Pvert_array
        Iidx = 0; Bidx = 0
        for Pvert in self.Pvert_array:
            if Pvert.type == 'internal':
                Pvert.set_Tidx(Iidx); Iidx += 1
            else:
                Pvert.set_Tidx(Bidx); Bidx += 1


        # Add all truncated indices to Dcell_array
        Iidx = 0; Bidx = 0
        for Dcell in self.Dcell_array:
            # if Dcell.type == 'internal':
            Dcell.set_Tidx(Iidx); Iidx += 1
            # else:
                # Dcell.set_Tidx(Bidx); Bidx += 1

        # Add all truncated indices to Dface_array
        Iidx = 0; Bidx = 0
        for Dface in self.Dface_array:
            if Dface.type == 'internal':
                Dface.set_Tidx(Iidx); Iidx += 1
            else:
                Dface.set_Tidx(Bidx); Bidx += 1

        # Add all truncated indices to Dvert_array
        Iidx = 0; Bidx = 0
        for Dvert in self.Dvert_array:
            # if Dvert.type == 'internal':
            Dvert.set_Tidx(Iidx); Iidx += 1
            # else:
            #     Dvert.set_Tidx(Bidx); Bidx += 1
        #
        # print([self.Dface_array[i].type for i in range(len(self.Dface_array))])

        # Add areas and cell-centers to Pcell_array entries:
        for Pcell in self.Pcell_array:
            NEidx = Pcell.vertices_idx[dir2D.NE]
            NWidx = Pcell.vertices_idx[dir2D.NW]
            SEidx = Pcell.vertices_idx[dir2D.SE]
            SWidx = Pcell.vertices_idx[dir2D.SW]

            if NEidx > -1: xNE, yNE = self.Pvert_array[NEidx].coordinates[dir2D.x], self.Pvert_array[NEidx].coordinates[dir2D.y]
            if NWidx > -1: xNW, yNW = self.Pvert_array[NWidx].coordinates[dir2D.x], self.Pvert_array[NWidx].coordinates[dir2D.y]
            if SEidx > -1: xSE, ySE = self.Pvert_array[SEidx].coordinates[dir2D.x], self.Pvert_array[SEidx].coordinates[dir2D.y]
            if SWidx > -1: xSW, ySW = self.Pvert_array[SWidx].coordinates[dir2D.x], self.Pvert_array[SWidx].coordinates[dir2D.y]

            if NEidx > -1 and NWidx > -1:
                Pcell.coordinates[dir2D.x] = (xNE + xNW)/2
                hx = abs(xNE-xNW)
            else:
                Pcell.coordinates[dir2D.x] = (xSE + xSW)/2
                hx = abs(xSE-xSW)

            if NEidx > -1 and SEidx > -1:
                Pcell.coordinates[dir2D.y] = (yNE + ySE)/2
                hy = abs(yNE-ySE)
            else:
                Pcell.coordinates[dir2D.y] = (yNW + ySW)/2
                hy = abs(yNW-ySW)
            Pcell.volume               = hx*hy

        # Add areas and cell-centers to Dcell_array entries:
        for Dcell in self.Dcell_array:
            NEidx = Dcell.vertices_idx[dir2D.NE]
            NWidx = Dcell.vertices_idx[dir2D.NW]
            SEidx = Dcell.vertices_idx[dir2D.SE]
            SWidx = Dcell.vertices_idx[dir2D.SW]

            if NEidx > -1: xNE, yNE = self.Dvert_array[NEidx].coordinates[dir2D.x], self.Dvert_array[NEidx].coordinates[dir2D.y]
            if NWidx > -1: xNW, yNW = self.Dvert_array[NWidx].coordinates[dir2D.x], self.Dvert_array[NWidx].coordinates[dir2D.y]
            if SEidx > -1: xSE, ySE = self.Dvert_array[SEidx].coordinates[dir2D.x], self.Dvert_array[SEidx].coordinates[dir2D.y]
            if SWidx > -1: xSW, ySW = self.Dvert_array[SWidx].coordinates[dir2D.x], self.Dvert_array[SWidx].coordinates[dir2D.y]

            if NEidx > -1 and NWidx > -1:
                Dcell.coordinates[dir2D.x] = (xNE + xNW)/2
                hx = abs(xNE-xNW)
            else:
                Dcell.coordinates[dir2D.x] = (xSE + xSW)/2
                hx = abs(xSE-xSW)

            if NEidx > -1 and SEidx > -1:
                Dcell.coordinates[dir2D.y] = (yNE + ySE)/2
                hy = abs(yNE-ySE)
            else:
                Dcell.coordinates[dir2D.y] = (yNW + ySW)/2
                hy = abs(yNW-ySW)
            Dcell.volume               = hx*hy

    def get_primal(self) -> tuple[npt.NDArray[Cell], npt.NDArray[Face], npt.NDArray[Vertex]]:
        return self.Pcell_array, self.Pface_array, self.Pvert_array

    def get_dual(self) -> tuple[npt.NDArray[Cell], npt.NDArray[Face], npt.NDArray[Vertex]]:
        return self.Dcell_array, self.Dface_array, self.Dvert_array

if __name__ == "__main__":
    N = 3
    M = 5

    xp_test = np.asarray([1,2,3,4,5,6])
    yp_test = np.asarray([1,2,3,4])

    xd_test = np.asarray([1,2,3,4,5,6,7])
    yd_test = np.asarray([1,2,3,4,5])
    grid = Grid2D(N,M, xp=xp_test,yp=yp_test,xd=xd_test,yd=yd_test)
    # print([(vert.idx, vert.faces_idx[dir2D.N], vert.faces_idx[dir2D.E], vert.faces_idx[dir2D.S], vert.faces_idx[dir2D.W]) for vert in grid.Pvert_array])

    # print([(vert.idx, vert.faces_idx[dir2D.N], vert.faces_idx[dir2D.E], vert.faces_idx[dir2D.S], vert.faces_idx[dir2D.W]) for vert in grid.Dvert_array])
    # print([face.Tidx for face in grid.Dface_array])
    # print([face.idx for face in grid.Dface_array])
    # print([face.type for face in grid.Dface_array])

    # print(grid.Pcell_array[16])
    # print(grid.Pface_array[16])
    # print(grid.Pvert_array[16])
    print([(Dcell.idx, Dcell.vertices_idx) for Dcell in grid.Dface_array if Dcell.type == 'north'])
    # print(grid.PCgrid)
    #print(grid.get_boundary_primal_faces_idx())
