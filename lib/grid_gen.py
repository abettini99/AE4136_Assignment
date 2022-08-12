#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
import numpy as np
import numpy.typing as npt
from math import nan
from dataclasses import dataclass, field
try:
    from utils import *
    from cell import *
    from face import *
    from vertex import *
except ModuleNotFoundError:
    from lib.utils import *
    from lib.cell import *
    from lib.face import *
    from lib.vertex import *

# Functions / Classes
@dataclass() # Not super necessary to have this decorator for this to work, but it is still nice to have.
class Grid2D:

    #### ============ ####
    #### Class inputs ####
    #### ============ ####
    N: int          ## Number of cells in the x-direction
    M: int          ## Number of cells in the y-direction
    type: str       ## Spacing type

    #### =================== ####
    #### Post-init variables ####
    #### =================== ####
    # Abbreviations
    # P     - primal
    # D     - dual
    # C     - cell
    # F     - face
    # V     - vertex
    Pcell_array: npt.NDArray[object]    = field(init=False, repr=False)
    Pface_array: npt.NDArray[object]    = field(init=False, repr=False)
    Pvert_array: npt.NDArray[object]    = field(init=False, repr=False)
    Dcell_array: npt.NDArray[object]    = field(init=False, repr=False)
    Dface_array: npt.NDArray[object]    = field(init=False, repr=False)
    Dvert_array: npt.NDArray[object]    = field(init=False, repr=False)

    xp:     npt.NDArray[np.float64]     = field(init=False, repr=False)   # grid points on primal grid
    xd:     npt.NDArray[np.float64]     = field(init=False, repr=False)   # grid points on dual grid
    hxp:    npt.NDArray[np.float64]     = field(init=False, repr=False)   # mesh width primal grid
    hxd:    npt.NDArray[np.float64]     = field(init=False, repr=False)   # mesh width dual grid
    yp:     npt.NDArray[np.float64]     = field(init=False, repr=False)   # grid points on primal grid
    yd:     npt.NDArray[np.float64]     = field(init=False, repr=False)   # grid points on dual grid
    hyp:    npt.NDArray[np.float64]     = field(init=False, repr=False)   # mesh width primal grid
    hyd:    npt.NDArray[np.float64]     = field(init=False, repr=False)   # mesh width dual grid

    hmin: float                         = field(init=False, repr=False)

    def __post_init__(self) -> None:

        #### ============================================================== ####
        ####                         Check input                            ####
        #### ============================================================== ####

        if self.type != 'cosine':
                raise ValueError(f'Grid created with unknown spacing type')

        #### ============================================================== ####
        ####                     Basic variable setup                       ####
        #### ============================================================== ####

        N: int = self.N; M: int = self.M
        L: int = 1
        idx: int; Iidx: int; Bidx: int; xidx: int; yidx: int

        ## Cosine setup
        self.xp     = np.zeros((M+1),         dtype = np.float64)   # grid points on primal grid
        self.xd     = np.zeros((M+2),         dtype = np.float64)   # grid points on dual grid
        self.hxp    = np.zeros((M),           dtype = np.float64)   # mesh width primal grid
        self.hxd    = np.zeros((M+1),         dtype = np.float64)   # mesh width dual grid
        self.yp     = np.zeros((N+1),         dtype = np.float64)   # grid points on primal grid
        self.yd     = np.zeros((N+2),         dtype = np.float64)   # grid points on dual grid
        self.hyp    = np.zeros((N),           dtype = np.float64)   # mesh width primal grid
        self.hyd    = np.zeros((N+1),         dtype = np.float64)   # mesh width dual grid

        if self.type == 'cosine':
            self.xp, self.xd, self.hxp, self.hxd = cosine_spacing(M, L)
            self.yp, self.yd, self.hyp, self.hyd = cosine_spacing(N, L)

        self.hmin = np.hstack( (self.hxp,self.hxd,self.hyp,self.hyd) ).min()

        #### ============================================================== ####
        ####                    Create primal domain                        ####
        #### ============================================================== ####

        PCgrid:     npt.NDArray[object] = np.empty((N+4,M+4),dtype=object)      # +2 for virtual cells, +2 extra for ghost cells (simplifies construction)
        PFgrid_EW:  npt.NDArray[object] = np.empty((N+4,M+5),dtype=object)      # +1 always compared to #virtual+real cells, +2 for virtual cells, +2 for ghost cells (not necessary, but simplifies)
        PFgrid_NS:  npt.NDArray[object] = np.empty((N+5,M+4),dtype=object)
        PVgrid:     npt.NDArray[object] = np.empty((N+5,M+5),dtype=object)      # +1 always compared to #virtual+real cells, +2 for virtual cells, +2 for ghost cells (not necessary, but simplifies)

        for i in range(N+4):
            for j in range(M+4):
                PCgrid[i,j] = Cell(2,'ghost')
        for i in range(N+4):
            for j in range(M+5):
                PFgrid_EW[i,j] = Face(2,'ghost')
        for i in range(N+5):
            for j in range(M+4):
                PFgrid_NS[i,j] = Face(2,'ghost')
        for i in range(N+5):
            for j in range(M+5):
                PVgrid[i,j] = Vertex(2,'ghost')

        #### ============================= ####
        #### Create cells, faces, vertices ####
        #### ============================= ####

        ## cells
        for i in range(1,N+3):
            for j in range(1,M+3):
                ## Setup virtual cells
                if (i == 1 or i == N+2) or (j == 1 or j == M+2):
                    if (i == 1 or i == N+2) and not (j == 1 or j == M+2):
                        PCgrid[i,j] = Cell(2,'virtualNS')
                    elif (j == 1 or j == M+2) and not (i == 1 or i == N+2):
                        PCgrid[i,j] = Cell(2,'virtualEW')
                ## Setup real cells
                else:
                    PCgrid[i,j] = Cell(2,'internal')

        ## faces
        for i in range(N+4):
            for j in range(M+4):
                ## Setup boundary faces
                if PCgrid[i,j].type == 'virtualNS':
                    PFgrid_NS[i,j]   = Face(2,'internal') if PCgrid[i-1,j].type == 'internal' else Face(2,'south')
                    PFgrid_NS[i+1,j] = Face(2,'internal') if PCgrid[i+1,j].type == 'internal' else Face(2,'north')
                elif PCgrid[i,j].type == 'virtualEW':
                    PFgrid_EW[i,j]   = Face(2,'internal') if PCgrid[i,j-1].type == 'internal' else Face(2,'west')
                    PFgrid_EW[i,j+1] = Face(2,'internal') if PCgrid[i,j+1].type == 'internal' else Face(2,'east')
                ## Setup real faces
                elif PCgrid[i,j].type == 'internal':
                    PFgrid_NS[i,j]   = Face(2,'internal')
                    PFgrid_NS[i+1,j] = Face(2,'internal')
                    PFgrid_EW[i,j]   = Face(2,'internal')
                    PFgrid_EW[i,j+1] = Face(2,'internal')

        # vertices
        for i in range(N+4):
            for j in range(M+4):
                ## Setup boundary vertices
                if PCgrid[i,j].type == 'virtualNS':
                    PVgrid[i,j]     = Vertex(2,'internal') if PCgrid[i-1,j].type == 'internal' else Vertex(2,'south')  # SW
                    PVgrid[i,j+1]   = Vertex(2,'internal') if PCgrid[i-1,j].type == 'internal' else Vertex(2,'south')  # SE
                    PVgrid[i+1,j+1] = Vertex(2,'internal') if PCgrid[i+1,j].type == 'internal' else Vertex(2,'north')  # NE
                    PVgrid[i+1,j]   = Vertex(2,'internal') if PCgrid[i+1,j].type == 'internal' else Vertex(2,'north')  # NW
                elif PCgrid[i,j].type == 'virtualEW':
                    PVgrid[i,j]     = Vertex(2,'internal') if PCgrid[i,j-1].type == 'internal' else Vertex(2,'west')  # SW
                    PVgrid[i+1,j]   = Vertex(2,'internal') if PCgrid[i,j-1].type == 'internal' else Vertex(2,'west')  # NW
                    PVgrid[i,j+1]   = Vertex(2,'internal') if PCgrid[i,j+1].type == 'internal' else Vertex(2,'east')  # SE
                    PVgrid[i+1,j+1] = Vertex(2,'internal') if PCgrid[i,j+1].type == 'internal' else Vertex(2,'east')  # NE
                ## Setup real vertices
                elif PCgrid[i,j].type == 'internal':
                    PVgrid[i,j]     = Vertex(2,'internal') # SW
                    PVgrid[i,j+1]   = Vertex(2,'internal') # SE
                    PVgrid[i+1,j+1] = Vertex(2,'internal') # NE
                    PVgrid[i+1,j]   = Vertex(2,'internal') # NW

        #### ================= ####
        #### Establish indices ####
        #### ================= ####

        ## cells
        idx = 0; Iidx = 0; Bidx = 0
        for i in range(N+4):
            for j in range(M+4):
                if not PCgrid[i,j].type == 'ghost':
                    PCgrid[i,j].set_idx(idx); idx += 1
                    if PCgrid[i,j].type == 'internal':      PCgrid[i,j].set_Tidx(Iidx); Iidx += 1
                    elif PCgrid[i,j].type == 'virtualNS':   PCgrid[i,j].set_Tidx(Bidx); Bidx += 1
                    elif PCgrid[i,j].type == 'virtualEW':   PCgrid[i,j].set_Tidx(Bidx); Bidx += 1

        ## faces
        idx = 0; Iidx = 0; Bidx = 0
        for i in range(N+4):
            for j in range(M+5):
                if not PFgrid_EW[i,j].type == 'ghost':
                    PFgrid_EW[i,j].set_idx(idx); idx += 1
                    if PFgrid_EW[i,j].type == 'internal':   PFgrid_EW[i,j].set_Tidx(Iidx); Iidx += 1
                    elif PFgrid_EW[i,j].type == 'east':     PFgrid_EW[i,j].set_Tidx(Bidx); Bidx += 1
                    elif PFgrid_EW[i,j].type == 'west':     PFgrid_EW[i,j].set_Tidx(Bidx); Bidx += 1
        for i in range(N+5):
            for j in range(M+4):
                if not PFgrid_NS[i,j].type == 'ghost':
                    PFgrid_NS[i,j].set_idx(idx); idx += 1
                    if PFgrid_NS[i,j].type == 'internal':   PFgrid_NS[i,j].set_Tidx(Iidx); Iidx += 1
                    elif PFgrid_NS[i,j].type == 'north':    PFgrid_NS[i,j].set_Tidx(Bidx); Bidx += 1
                    elif PFgrid_NS[i,j].type == 'south':    PFgrid_NS[i,j].set_Tidx(Bidx); Bidx += 1

        ## vertices
        idx = 0; Iidx = 0; Bidx = 0
        for i in range(N+5):
            for j in range(M+5):
                if not PVgrid[i,j].type == 'ghost':
                    PVgrid[i,j].set_idx(idx); idx += 1
                    if PVgrid[i,j].type == 'internal':      PVgrid[i,j].set_Tidx(Iidx); Iidx += 1
                    elif PVgrid[i,j].type == 'north':       PVgrid[i,j].set_Tidx(Bidx); Bidx += 1
                    elif PVgrid[i,j].type == 'east':        PVgrid[i,j].set_Tidx(Bidx); Bidx += 1
                    elif PVgrid[i,j].type == 'south':       PVgrid[i,j].set_Tidx(Bidx); Bidx += 1
                    elif PVgrid[i,j].type == 'west':        PVgrid[i,j].set_Tidx(Bidx); Bidx += 1

        #### ===================== ####
        #### Establish connections ####
        #### ===================== ####
        for i in range(N+4):
            for j in range(M+4):
                if PCgrid[i,j].type != 'ghost':

                    # cell-cell connections
                    PCgrid[i,j].cells_idx[dir2D.N]      = PCgrid[i+1,j].idx
                    PCgrid[i,j].cells_idx[dir2D.E]      = PCgrid[i,j+1].idx
                    PCgrid[i,j].cells_idx[dir2D.S]      = PCgrid[i-1,j].idx
                    PCgrid[i,j].cells_idx[dir2D.W]      = PCgrid[i,j-1].idx

                    # cell-face connections
                    PCgrid[i,j].faces_idx[dir2D.N]      = PFgrid_NS[i+1,j].idx
                    PCgrid[i,j].faces_idx[dir2D.E]      = PFgrid_EW[i,j+1].idx
                    PCgrid[i,j].faces_idx[dir2D.S]      = PFgrid_NS[i,j].idx
                    PCgrid[i,j].faces_idx[dir2D.W]      = PFgrid_EW[i,j].idx

                    # face-cell connections:
                    PFgrid_NS[i+1,j].cells_idx[dir2D.Bp] = PCgrid[i,j].idx
                    PFgrid_EW[i,j+1].cells_idx[dir2D.Lp] = PCgrid[i,j].idx
                    PFgrid_NS[i,j].cells_idx[dir2D.Tp]   = PCgrid[i,j].idx
                    PFgrid_EW[i,j].cells_idx[dir2D.Rp]   = PCgrid[i,j].idx

                    # cell-vertex connections:
                    PCgrid[i,j].vertices_idx[dir2D.SW]  = PVgrid[i,j].idx
                    PCgrid[i,j].vertices_idx[dir2D.SE]  = PVgrid[i,j+1].idx
                    PCgrid[i,j].vertices_idx[dir2D.NE]  = PVgrid[i+1,j+1].idx
                    PCgrid[i,j].vertices_idx[dir2D.NW]  = PVgrid[i+1,j].idx

                    # vertex-cell connections:
                    PVgrid[i,j].cells_idx[dir2D.NE]     = PCgrid[i,j].idx
                    PVgrid[i,j+1].cells_idx[dir2D.NW]   = PCgrid[i,j].idx
                    PVgrid[i+1,j+1].cells_idx[dir2D.SW] = PCgrid[i,j].idx
                    PVgrid[i+1,j].cells_idx[dir2D.SE]   = PCgrid[i,j].idx

                    # face-vertex connections:
                    PFgrid_NS[i+1,j].vertices_idx[dir2D.Lp] = PVgrid[i+1,j].idx
                    PFgrid_NS[i+1,j].vertices_idx[dir2D.Rp] = PVgrid[i+1,j+1].idx
                    PFgrid_EW[i,j+1].vertices_idx[dir2D.Bp] = PVgrid[i,j+1].idx
                    PFgrid_EW[i,j+1].vertices_idx[dir2D.Tp] = PVgrid[i+1,j+1].idx
                    PFgrid_NS[i,j].vertices_idx[dir2D.Lp]   = PVgrid[i,j].idx
                    PFgrid_NS[i,j].vertices_idx[dir2D.Rp]   = PVgrid[i,j+1].idx
                    PFgrid_EW[i,j].vertices_idx[dir2D.Bp]   = PVgrid[i,j].idx
                    PFgrid_EW[i,j].vertices_idx[dir2D.Tp]   = PVgrid[i+1,j].idx

                    # vertex-face connections:
                    PVgrid[i,j].faces_idx[dir2D.N]      = PFgrid_EW[i,j].idx
                    PVgrid[i,j].faces_idx[dir2D.E]      = PFgrid_NS[i,j].idx
                    PVgrid[i,j+1].faces_idx[dir2D.N]    = PFgrid_EW[i,j+1].idx
                    PVgrid[i,j+1].faces_idx[dir2D.W]    = PFgrid_NS[i,j].idx
                    PVgrid[i+1,j+1].faces_idx[dir2D.S]  = PFgrid_EW[i,j+1].idx
                    PVgrid[i+1,j+1].faces_idx[dir2D.W]  = PFgrid_NS[i+1,j].idx
                    PVgrid[i+1,j].faces_idx[dir2D.S]    = PFgrid_EW[i,j].idx
                    PVgrid[i+1,j].faces_idx[dir2D.E]    = PFgrid_NS[i+1,j].idx

        #### ===================== ####
        #### Establish coordinates ####
        #### ===================== ####

        ## Place in x-coordinates
        xidx = 0
        for i in range(N+4):
            for j in range(M+4):
                if PCgrid[i,j].type != 'ghost':
                    if PCgrid[i,j].type == 'internal':
                        # Quite a lot of overlap, disregard.
                        PVgrid[i,j].coordinates[dir2D.x] = self.xp[xidx]
                        PVgrid[i,j+1].coordinates[dir2D.x] = self.xp[xidx+1]
                        PVgrid[i+1,j+1].coordinates[dir2D.x] = self.xp[xidx+1]
                        PVgrid[i+1,j].coordinates[dir2D.x] = self.xp[xidx]

                        PCgrid[i,j].h[dir2D.x] = self.hxp[xidx]

                        xidx += 1
                    elif PCgrid[i,j].type == 'virtualEW':
                        PVgrid[i,j].coordinates[dir2D.x] = self.xp[0] if PCgrid[i,j-1].type == 'ghost' else self.xp[-1]
                        PVgrid[i,j+1].coordinates[dir2D.x] = self.xp[0] if PCgrid[i,j-1].type == 'ghost' else self.xp[-1]
                        PVgrid[i+1,j+1].coordinates[dir2D.x] = self.xp[0] if PCgrid[i,j-1].type == 'ghost' else self.xp[-1]
                        PVgrid[i+1,j].coordinates[dir2D.x] = self.xp[0] if PCgrid[i,j-1].type == 'ghost' else self.xp[-1]

                        PCgrid[i,j].h[dir2D.x] = 0

                    elif PCgrid[i,j].type == 'virtualNS':
                        PVgrid[i,j].coordinates[dir2D.x] = self.xp[xidx]
                        PVgrid[i,j+1].coordinates[dir2D.x] = self.xp[xidx+1]
                        PVgrid[i+1,j+1].coordinates[dir2D.x] = self.xp[xidx+1]
                        PVgrid[i+1,j].coordinates[dir2D.x] = self.xp[xidx]

                        PCgrid[i,j].h[dir2D.x] = self.hxp[xidx]

                        xidx += 1
            xidx = 0

        ## Place in y-coordinates
        yidx = 0
        for j in range(M+4):
            for i in range(N+4):
                if PCgrid[i,j].type != 'ghost':
                    if PCgrid[i,j].type == 'internal':
                        # Quite a lot of overlap, disregard.
                        PVgrid[i,j].coordinates[dir2D.y] = self.yp[yidx]
                        PVgrid[i,j+1].coordinates[dir2D.y] = self.yp[yidx]
                        PVgrid[i+1,j+1].coordinates[dir2D.y] = self.yp[yidx+1]
                        PVgrid[i+1,j].coordinates[dir2D.y] = self.yp[yidx+1]

                        PCgrid[i,j].h[dir2D.y] = self.hyp[yidx]

                        yidx += 1
                    elif PCgrid[i,j].type == 'virtualEW':
                        PVgrid[i,j].coordinates[dir2D.y] = self.yp[yidx]
                        PVgrid[i,j+1].coordinates[dir2D.y] = self.yp[yidx]
                        PVgrid[i+1,j+1].coordinates[dir2D.y] = self.yp[yidx+1]
                        PVgrid[i+1,j].coordinates[dir2D.y] = self.yp[yidx+1]

                        PCgrid[i,j].h[dir2D.y] = self.hyp[yidx]

                        yidx += 1
                    elif PCgrid[i,j].type == 'virtualNS':
                        PVgrid[i,j].coordinates[dir2D.y] = self.yp[0] if PCgrid[i-1,j].type == 'ghost' else self.yp[-1]
                        PVgrid[i,j+1].coordinates[dir2D.y] = self.yp[0] if PCgrid[i-1,j].type == 'ghost' else self.yp[-1]
                        PVgrid[i+1,j+1].coordinates[dir2D.y] = self.yp[0] if PCgrid[i-1,j].type == 'ghost' else self.yp[-1]
                        PVgrid[i+1,j].coordinates[dir2D.y] = self.yp[0] if PCgrid[i-1,j].type == 'ghost' else self.yp[-1]

                        PCgrid[i,j].h[dir2D.y] = 0
            yidx = 0

        ## place in volumes
        for i in range(N+4):
            for j in range(M+4):
                if PCgrid[i,j].type != 'ghost':
                    PCgrid[i,j].volume = PCgrid[i,j].h[dir2D.x]*PCgrid[i,j].h[dir2D.y]

        #### Visualize cell grid
        # tmp = -np.ones((N+4,M+4))#,dtype=object)
        # for i in range(N+4):
        #     for j in range(M+4):
        #         tmp[i,j] = PCgrid[i,j].volume
        # print(tmp)

        #### Visualize face grid
        # tmp = -np.ones((N+4,M+5),dtype=object)
        # for i in range(N+4):
        #     for j in range(M+5):
        #         tmp[i,j] = PFgrid_EW[i,j].Tidx
        # print(tmp)
        # tmp = -np.ones((N+5,M+4),dtype=object)
        # for i in range(N+5):
        #     for j in range(M+4):
        #         tmp[i,j] = PFgrid_NS[i,j].Tidx
        # print(tmp)

        #### Visualize vertex grid
        # tmp = -np.ones((N+5,M+5))#,dtype=object)
        # for i in range(N+5):
        #     for j in range(M+5):
        #         tmp[i,j] = PVgrid[i,j].idx
        # print(tmp)

        #### ============================================================== ####
        ####                      Create dual domain                        ####
        #### ============================================================== ####

        DCgrid:     npt.NDArray[object] = np.empty((N+3,M+3),dtype=object)      # +1 always compared to primal grid, +2 extra for ghost cells (simplifies construction)
        DFgrid_EW:  npt.NDArray[object] = np.empty((N+3,M+4),dtype=object)      # +1 always compared to #virtual+real cells, +1 always compared to primal faces, +2 for ghost cells (not necessary, but simplifies)
        DFgrid_NS:  npt.NDArray[object] = np.empty((N+4,M+3),dtype=object)
        DVgrid:     npt.NDArray[object] = np.empty((N+4,M+4),dtype=object)      # +1 always compared to #virtual+real cells, +1 always compared to primal faces, +2 for ghost cells (not necessary, but simplifies)

        for i in range(N+3):
            for j in range(M+3):
                DCgrid[i,j] = Cell(2,'ghost')
        for i in range(N+3):
            for j in range(M+4):
                DFgrid_EW[i,j] = Face(2,'ghost')
        for i in range(N+4):
            for j in range(M+3):
                DFgrid_NS[i,j] = Face(2,'ghost')
        for i in range(N+4):
            for j in range(M+4):
                DVgrid[i,j] = Vertex(2,'ghost')

        #### ============================= ####
        #### Create cells, faces, vertices ####
        #### ============================= ####

        ## cells
        for i in range(1,N+2):
            for j in range(1,M+2):
                ## Setup real cells
                DCgrid[i,j] = Cell(2,'internal')

        ## faces
        for i in range(N+3):
            for j in range(M+3):
                ## Setup boundary faces
                if DCgrid[i,j].type == 'internal':
                    DFgrid_NS[i,j]   = Face(2,'internal') if DCgrid[i-1,j].type == 'internal' else Face(2,'south')
                    DFgrid_NS[i+1,j] = Face(2,'internal') if DCgrid[i+1,j].type == 'internal' else Face(2,'north')
                    DFgrid_EW[i,j]   = Face(2,'internal') if DCgrid[i,j-1].type == 'internal' else Face(2,'west')
                    DFgrid_EW[i,j+1] = Face(2,'internal') if DCgrid[i,j+1].type == 'internal' else Face(2,'east')

        # vertices
        for i in range(N+3):
            for j in range(M+3):
                ## Setup real vertices
                if DCgrid[i,j].type == 'internal':
                    DVgrid[i,j]     = Vertex(2,'internal') if (DCgrid[i-1,j].type != 'ghost' or DCgrid[i,j-1].type != 'ghost') else Vertex(2,'virtual')  # SW
                    DVgrid[i,j+1]   = Vertex(2,'internal') if (DCgrid[i-1,j].type != 'ghost' or DCgrid[i,j+1].type != 'ghost') else Vertex(2,'virtual')  # SE
                    DVgrid[i+1,j+1] = Vertex(2,'internal') if (DCgrid[i+1,j].type != 'ghost' or DCgrid[i,j+1].type != 'ghost') else Vertex(2,'virtual')  # NE
                    DVgrid[i+1,j]   = Vertex(2,'internal') if (DCgrid[i+1,j].type != 'ghost' or DCgrid[i,j-1].type != 'ghost') else Vertex(2,'virtual')  # NW

        #### ================= ####
        #### Establish indices ####
        #### ================= ####

        ## cells
        idx = 0; Iidx = 0; Bidx = 0
        for i in range(N+3):
            for j in range(M+3):
                if not DCgrid[i,j].type == 'ghost':
                    DCgrid[i,j].set_idx(idx); idx += 1
                    if DCgrid[i,j].type == 'internal':      DCgrid[i,j].set_Tidx(Iidx); Iidx += 1

        ## faces, note that they are x-fluxes first, then y-fluxes
        idx = 0; Iidx = 0; Bidx = 0
        for i in range(N+4):
            for j in range(M+3):
                if not DFgrid_NS[i,j].type == 'ghost':
                    DFgrid_NS[i,j].set_idx(idx); idx += 1
                    if DFgrid_NS[i,j].type == 'internal':   DFgrid_NS[i,j].set_Tidx(Iidx); Iidx += 1
                    elif DFgrid_NS[i,j].type == 'north':    DFgrid_NS[i,j].set_Tidx(Bidx); Bidx += 1
                    elif DFgrid_NS[i,j].type == 'south':    DFgrid_NS[i,j].set_Tidx(Bidx); Bidx += 1
        for i in range(N+3):
            for j in range(M+4):
                if not DFgrid_EW[i,j].type == 'ghost':
                    DFgrid_EW[i,j].set_idx(idx); idx += 1
                    if DFgrid_EW[i,j].type == 'internal':   DFgrid_EW[i,j].set_Tidx(Iidx); Iidx += 1
                    elif DFgrid_EW[i,j].type == 'east':     DFgrid_EW[i,j].set_Tidx(Bidx); Bidx += 1
                    elif DFgrid_EW[i,j].type == 'west':     DFgrid_EW[i,j].set_Tidx(Bidx); Bidx += 1

        ## vertices
        idx = 0; Iidx = 0; Bidx = 0
        for i in range(N+4):
            for j in range(M+4):
                if not DVgrid[i,j].type == 'ghost':
                    DVgrid[i,j].set_idx(idx); idx += 1
                    if DVgrid[i,j].type == 'internal':      DVgrid[i,j].set_Tidx(Iidx); Iidx += 1
                    elif DVgrid[i,j].type == 'virtual':     DVgrid[i,j].set_Tidx(Bidx); Bidx += 1

        #### ===================== ####
        #### Establish connections ####
        #### ===================== ####
        for i in range(N+3):
            for j in range(M+3):
                if DCgrid[i,j].type != 'ghost':

                    # cell-cell connections
                    DCgrid[i,j].cells_idx[dir2D.N]      = DCgrid[i+1,j].idx
                    DCgrid[i,j].cells_idx[dir2D.E]      = DCgrid[i,j+1].idx
                    DCgrid[i,j].cells_idx[dir2D.S]      = DCgrid[i-1,j].idx
                    DCgrid[i,j].cells_idx[dir2D.W]      = DCgrid[i,j-1].idx

                    # cell-face connections
                    DCgrid[i,j].faces_idx[dir2D.N]      = DFgrid_NS[i+1,j].idx
                    DCgrid[i,j].faces_idx[dir2D.E]      = DFgrid_EW[i,j+1].idx
                    DCgrid[i,j].faces_idx[dir2D.S]      = DFgrid_NS[i,j].idx
                    DCgrid[i,j].faces_idx[dir2D.W]      = DFgrid_EW[i,j].idx

                    # face-cell connections:
                    DFgrid_NS[i+1,j].cells_idx[dir2D.Bd] = DCgrid[i,j].idx
                    DFgrid_EW[i,j+1].cells_idx[dir2D.Ld] = DCgrid[i,j].idx
                    DFgrid_NS[i,j].cells_idx[dir2D.Td]   = DCgrid[i,j].idx
                    DFgrid_EW[i,j].cells_idx[dir2D.Rd]   = DCgrid[i,j].idx

                    # cell-vertex connections:
                    DCgrid[i,j].vertices_idx[dir2D.SW]  = DVgrid[i,j].idx
                    DCgrid[i,j].vertices_idx[dir2D.SE]  = DVgrid[i,j+1].idx
                    DCgrid[i,j].vertices_idx[dir2D.NE]  = DVgrid[i+1,j+1].idx
                    DCgrid[i,j].vertices_idx[dir2D.NW]  = DVgrid[i+1,j].idx

                    # vertex-cell connections:
                    DVgrid[i,j].cells_idx[dir2D.NE]     = DCgrid[i,j].idx
                    DVgrid[i,j+1].cells_idx[dir2D.NW]   = DCgrid[i,j].idx
                    DVgrid[i+1,j+1].cells_idx[dir2D.SW] = DCgrid[i,j].idx
                    DVgrid[i+1,j].cells_idx[dir2D.SE]   = DCgrid[i,j].idx

                    # face-vertex connections:
                    DFgrid_NS[i+1,j].vertices_idx[dir2D.Ld] = DVgrid[i+1,j].idx
                    DFgrid_NS[i+1,j].vertices_idx[dir2D.Rd] = DVgrid[i+1,j+1].idx
                    DFgrid_EW[i,j+1].vertices_idx[dir2D.Bd] = DVgrid[i,j+1].idx
                    DFgrid_EW[i,j+1].vertices_idx[dir2D.Td] = DVgrid[i+1,j+1].idx
                    DFgrid_NS[i,j].vertices_idx[dir2D.Ld]   = DVgrid[i,j].idx
                    DFgrid_NS[i,j].vertices_idx[dir2D.Rd]   = DVgrid[i,j+1].idx
                    DFgrid_EW[i,j].vertices_idx[dir2D.Bd]   = DVgrid[i,j].idx
                    DFgrid_EW[i,j].vertices_idx[dir2D.Td]   = DVgrid[i+1,j].idx

                    # vertex-face connections:
                    DVgrid[i,j].faces_idx[dir2D.N]      = DFgrid_EW[i,j].idx
                    DVgrid[i,j].faces_idx[dir2D.E]      = DFgrid_NS[i,j].idx
                    DVgrid[i,j+1].faces_idx[dir2D.N]    = DFgrid_EW[i,j+1].idx
                    DVgrid[i,j+1].faces_idx[dir2D.W]    = DFgrid_NS[i,j].idx
                    DVgrid[i+1,j+1].faces_idx[dir2D.S]  = DFgrid_EW[i,j+1].idx
                    DVgrid[i+1,j+1].faces_idx[dir2D.W]  = DFgrid_NS[i+1,j].idx
                    DVgrid[i+1,j].faces_idx[dir2D.S]    = DFgrid_EW[i,j].idx
                    DVgrid[i+1,j].faces_idx[dir2D.E]    = DFgrid_NS[i+1,j].idx

        #### ===================== ####
        #### Establish coordinates ####
        #### ===================== ####

        ## Place in x-coordinates
        xidx = 0
        for i in range(N+3):
            for j in range(M+3):
                if DCgrid[i,j].type != 'ghost':
                    if DCgrid[i,j].type == 'internal' or DCgrid[i,j].type == 'virtual':
                        # Quite a lot of overlap, disregard.
                        DVgrid[i,j].coordinates[dir2D.x] = self.xd[xidx]
                        DVgrid[i,j+1].coordinates[dir2D.x] = self.xd[xidx+1]
                        DVgrid[i+1,j+1].coordinates[dir2D.x] = self.xd[xidx+1]
                        DVgrid[i+1,j].coordinates[dir2D.x] = self.xd[xidx]

                        DCgrid[i,j].h[dir2D.x] = self.hxd[xidx]

                        xidx += 1
            xidx = 0

        ## Place in y-coordinates
        yidx = 0
        for j in range(M+3):
            for i in range(N+3):
                if DCgrid[i,j].type != 'ghost':
                    if DCgrid[i,j].type == 'internal' or DCgrid[i,j].type == 'virtual':
                        # Quite a lot of overlap, disregard.
                        DVgrid[i,j].coordinates[dir2D.y] = self.yd[yidx]
                        DVgrid[i,j+1].coordinates[dir2D.y] = self.yd[yidx]
                        DVgrid[i+1,j+1].coordinates[dir2D.y] = self.yd[yidx+1]
                        DVgrid[i+1,j].coordinates[dir2D.y] = self.yd[yidx+1]

                        DCgrid[i,j].h[dir2D.y] = self.hyd[yidx]

                        yidx += 1
            yidx = 0

        ## place in volumes
        for i in range(N+3):
            for j in range(M+3):
                if DCgrid[i,j].type != 'ghost':
                    DCgrid[i,j].volume = DCgrid[i,j].h[dir2D.x]*DCgrid[i,j].h[dir2D.y]

        #### Visualize cell grid
        # tmp = -np.ones((N+3,M+3))#,dtype=object)
        # for i in range(N+3):
        #     for j in range(M+3):
        #         tmp[i,j] = DCgrid[i,j].Tidx
        # print(tmp)

        #### Visualize face grid
        # tmp = -np.ones((N+3,M+4))#,dtype=object)
        # for i in range(N+3):
        #     for j in range(M+4):
        #         tmp[i,j] = DFgrid_EW[i,j].Tidx
        # print(tmp)
        # tmp = -np.ones((N+4,M+3))#,dtype=object)
        # for i in range(N+4):
        #     for j in range(M+3):
        #         tmp[i,j] = DFgrid_NS[i,j].Tidx
        # print(tmp)

        #### Visualize vertex grid
        # tmp = -np.ones((N+4,M+4))#,dtype=object)
        # for i in range(N+4):
        #     for j in range(M+4):
        #         tmp[i,j] = DVgrid[i,j].coordinates[dir2D.y]
        # print(tmp)

        #### ============================================================== ####
        ####                    establish Hodge indices                     ####
        #### ============================================================== ####

        ## face-face Hodge indices, see fig 3 of assignment
        # -1 to x, -1 to y due to DF grid being one unit smaller than PF grid
        for i in range(N+4):
            for j in range(M+4):
                if PCgrid[i,j].type == 'internal':
                    id, jd = i-1, j-1

                    # primal-dual
                    PFgrid_EW[i,j].Midx   = DFgrid_NS[i,jd].idx
                    PFgrid_EW[i,j+1].Midx = DFgrid_NS[i,jd+1].idx
                    PFgrid_NS[i,j].Midx   = DFgrid_EW[id,j].idx
                    PFgrid_NS[i+1,j].Midx = DFgrid_EW[id+1,j].idx

                    # dual-primal
                    DFgrid_NS[i,jd].Midx     = PFgrid_EW[i,j].idx
                    DFgrid_NS[i,jd+1].Midx   = PFgrid_EW[i,j+1].idx
                    DFgrid_EW[id,j].Midx     = PFgrid_NS[i,j].idx
                    DFgrid_EW[id+1,j].Midx   = PFgrid_NS[i+1,j].idx

        ## vertex-cell Hodge indices
        # -1 to x, -1 to y due to DCgrid being one unit smaller than PCgrid
        for i in range(N+4):
            for j in range(M+4):
                if PCgrid[i,j].type == 'internal':
                    id, jd = i-1, j-1

                    # primal-dual
                    PVgrid[i,j].Midx        = DCgrid[id,jd].idx
                    PVgrid[i,j+1].Midx      = DCgrid[id,jd+1].idx
                    PVgrid[i+1,j+1].Midx    = DCgrid[id+1,jd+1].idx
                    PVgrid[i+1,j].Midx      = DCgrid[id+1,jd].idx

                    # dual-primal
                    DCgrid[id,jd].Midx      = PVgrid[i,j].idx
                    DCgrid[id,jd+1].Midx    = PVgrid[i,j+1].idx
                    DCgrid[id+1,jd+1].Midx  = PVgrid[i+1,j+1].idx
                    DCgrid[id+1,jd].Midx    = PVgrid[i+1,j].idx

        #### ============================================================== ####
        ####                        finalize domains                        ####
        #### ============================================================== ####

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


    def get_primal(self) -> tuple[npt.NDArray[object], npt.NDArray[object], npt.NDArray[object]]:
        return self.Pcell_array, self.Pface_array, self.Pvert_array

    def get_dual(self) -> tuple[npt.NDArray[object], npt.NDArray[object], npt.NDArray[object]]:
        return self.Dcell_array, self.Dface_array, self.Dvert_array

    def scale_x_domain(self, L: float) -> None:

        # Scale returnable arrays
        self.xp     *= L
        self.xd     *= L
        self.hxp    *= L
        self.hxd    *= L

        # Check minimum length requirement again
        self.hmin = np.hstack( (self.hxp,self.hxd,self.hyp,self.hyd) ).min()

        # Scale all internal points
        for cell in self.Pcell_array:
            cell.volume *= L
            cell.h[dir2D.x] *= L
            cell.coordinates[dir2D.x] *= L if cell.coordinates[dir2D.x] != nan else nan
        for vertex in self.Pvert_array:
            vertex.coordinates[dir2D.x] *= L if vertex.coordinates[dir2D.x] != nan else nan

        for cell in self.Dcell_array:
            cell.volume *= L
            cell.h[dir2D.x] *= L
            cell.coordinates[dir2D.x] *= L if cell.coordinates[dir2D.x] != nan else nan
        for vertex in self.Dvert_array:
            vertex.coordinates[dir2D.x] *= L if vertex.coordinates[dir2D.x] != nan else nan

    def scale_y_domain(self, L: float) -> None:

        # Scale returnable arrays
        self.yp     *= L
        self.yd     *= L
        self.hyp    *= L
        self.hyd    *= L

        # Check minimum length requirement again
        self.hmin = np.hstack( (self.hxp,self.hxd,self.hyp,self.hyd) ).min()

        # Scale all internal points
        for cell in self.Pcell_array:
            cell.volume *= L
            cell.h[dir2D.y] *= L
            cell.coordinates[dir2D.y] *= L if cell.coordinates[dir2D.y] != nan else nan
        for vertex in self.Pvert_array:
            vertex.coordinates[dir2D.y] *= L if vertex.coordinates[dir2D.y] != nan else nan

        for cell in self.Dcell_array:
            cell.volume *= L
            cell.h[dir2D.y] *= L
            cell.coordinates[dir2D.y] *= L if cell.coordinates[dir2D.y] != nan else nan
        for vertex in self.Dvert_array:
            vertex.coordinates[dir2D.y] *= L if vertex.coordinates[dir2D.y] != nan else nan

    def get_primal_lengths(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.xp, self.yp, self.hxp, self.hyp

    def get_dual_lengths(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.xd, self.yd, self.hxd, self.hyd

if __name__ == "__main__":
    N = 3
    M = 3

    grid: Grid2D = Grid2D(N,M,'cosine')

    grid.scale_x_domain(2.2)
    grid.scale_y_domain(2.1)
    # print([(vert.idx, vert.faces_idx) for vert in grid.Dvert_array])

    # print([(vert.idx, vert.faces_idx[dir2D.N], vert.faces_idx[dir2D.E], vert.faces_idx[dir2D.S], vert.faces_idx[dir2D.W]) for vert in grid.Dvert_array])
    # print([face.Tidx for face in grid.Dface_array])
    # print([face.idx for face in grid.Dface_array])
    # print([face.type for face in grid.Dface_array])

    # print(grid.Pcell_array[16])
    # print(grid.Pface_array[16])
    # print(grid.Pvert_array[16])
    # print([(Dcell.idx, Dcell.vertices_idx) for Dcell in grid.Dface_array if Dcell.type == 'north'])
    # print(grid.PCgrid)
    #print(grid.get_boundary_primal_faces_idx())
