#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
from dataclasses import dataclass, field
from math import nan
# from copy import deepcopy

# Functions / Classes
@dataclass(order=True) # Not super necessary to have this decorator for this to work, but it is still nice to have.
class Cell:

    #### ============ ####
    #### Class inputs ####
    #### ============ ####
    dim:  int
    type: str

    #### =================== ####
    #### Post-init variables ####
    #### =================== ####
    sort_index:     int         = field(init=False, repr=False)
    idx:            int         = field(init=False, repr=False)
    faces_idx:      list[int]   = field(init=False, repr=False)
    cells_idx:      list[int]   = field(init=False, repr=False)
    vertices_idx:   list[int]   = field(init=False, repr=False)

    # truncated indices -> indices used when constructing with truncated solution set, e.g. no boundaries
    Tidx:           int         = field(init=False, repr=False)
    # Tneighbours_idx:list[int]   = field(init=False, repr=False)
    # Tfaces_idx:     list[int]   = field(init=False, repr=False)
    # Tvertices_idx:  list[int]   = field(init=False, repr=False)

    # mirror indices -> index that face intersects with on the mirror mesh (i.e. dual edge intersection with primal edge)
    Midx:           int         = field(init=False, repr=False)
    
    coordinates:    list[float] = field(init=False, repr=False)
    volume:         float       = field(init=False, repr=False)
    h:              list[float] = field(init=False, repr=False)
    U:              list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:

        "Default configuration for cells initially imply rectangular (prism) construction, can be overwritten"

        self.idx            = -1
        self.sort_index     = self.idx
        self.U              = [0]
        self.volume         = nan

        if self.dim == 1:
            self.cells_idx      = [-1,-1]
            self.faces_idx      = [-1,-1]
            self.vertices_idx   = [-1,-1]
            self.coordinates    = [0]
            self.h              = [nan]
        elif self.dim == 2:
            self.cells_idx      = [-1,-1,-1,-1]
            self.faces_idx      = [-1,-1,-1,-1]
            self.vertices_idx   = [-1,-1,-1,-1]
            self.coordinates    = [nan,nan]
            self.h              = [nan,nan]
        elif self.dim == 3:
            self.cells_idx      = [-1,-1,-1,-1,-1,-1]
            self.faces_idx      = [-1,-1,-1,-1,-1,-1]
            self.vertices_idx   = [-1,-1,-1,-1,-1,-1]
            self.coordinates    = [nan,nan,nan]
            self.h              = [nan,nan,nan]
        else:
            raise ValueError(f'Cell created with incorrect dimensions')

        self.Tidx               = -1
        # self.Tneighbours_idx    = deepcopy(self.neighbours_idx)
        # self.Tfaces_idx         = deepcopy(self.faces_idx)
        # self.Tvertices_idx      = deepcopy(self.vertices_idx)

    def set_idx(self, idx: int) -> None:
        self.idx = idx
        self.sort_index = self.idx

    def set_Tidx(self, Tidx: int) -> None:
        self.Tidx = Tidx

    def __str__(self) -> str:
        return  f'{self.dim}D Cell: {self.type}\n'+ \
                f'idx, Tidx: {self.idx}, {self.Tidx}\n'+ \
                f'faces = {self.faces_idx}\n' + \
                f'neighbours = {self.cells_idx}\n'+ \
                f'vertices = {self.vertices_idx}\n'+ \
                f'coordinates = {[round(coordinate,3) for coordinate in self.coordinates]}\n'+ \
                f'volume = {round(self.volume,3)}\n'+ \
                f'U = {[round(u,3) for u in (self.U)]}\n'

if __name__ == "__main__":

    cell = Cell(2,'r')
    print(cell)
    # print(cell.faces_idx)

    cell.faces_idx = [-1,-1,-1]
    print(cell)
    # print(cell.faces_idx)

    # Cell(4,'r')
