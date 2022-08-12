#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
from dataclasses import dataclass, field
from math import nan
# from copy import deepcopy

# Functions / Classes
@dataclass(order=True) # Not super necessary to have this decorator for this to work, but it is still nice to have.
class Vertex:

    #### ============ ####
    #### Class inputs ####
    #### ============ ####
    dim: int
    type: str

    #### =================== ####
    #### Post-init variables ####
    #### =================== ####
    sort_index:     int         = field(init=False, repr=False)
    idx:            int         = field(init=False, repr=False)
    cells_idx:      list[int]   = field(init=False, repr=False)
    faces_idx:      list[int]   = field(init=False, repr=False)

    # truncated indices -> indices used when constructing with truncated solution set, e.g. no boundaries
    Tidx:           int         = field(init=False, repr=False)
    # Tcells_idx:     list[int]   = field(init=False, repr=False)
    # Tfaces_idx:     list[int]   = field(init=False, repr=False)

    # mirror indices -> index that face intersects with on the mirror mesh (i.e. dual edge intersection with primal edge)
    Midx:           int         = field(init=False, repr=False)

    def __post_init__(self) -> None:

        "Default configuration for vertices initially imply rectangular (prism) construction, can be overwritten"

        self.idx            = -1
        self.sort_index     = self.idx

        if self.dim == 1:
            raise ValueError(f'One dimensional vertices do not exist. Use the Face() class instead.')
        elif self.dim == 2:
            self.cells_idx      = [-1,-1,-1,-1]
            self.faces_idx      = [-1,-1,-1,-1]
            self.coordinates    = [nan,nan]
        elif self.dim == 3:
            self.cells_idx      = [-1,-1,-1,-1,-1,-1,-1,-1]
            self.faces_idx      = [-1,-1,-1,-1,-1,-1]
            self.coordinates    = [nan,nan,nan]
        else:
            raise ValueError(f'Vertex created with incorrect dimensions')

        self.Tidx           = -1
        # self.Tcells_idx     = deepcopy(self.cells_idx)
        # self.Tfaces_idx     = deepcopy(self.faces_idx)

    def set_idx(self, idx: int) -> None:
        self.idx = idx
        self.sort_index = self.idx

    def set_Tidx(self, Tidx: int) -> None:
        self.Tidx = Tidx

    def __str__(self) -> str:
        return  f'{self.dim}D Vertex: {self.type}\n'+ \
                f'idx, Tidx: {self.idx}, {self.Tidx}\n'+ \
                f'cell neighbours = {self.cells_idx}\n'+ \
                f'face neighbours = {self.faces_idx}\n'+ \
                f'coordinates = {[round(coordinate,3) for coordinate in self.coordinates]}\n'

if __name__ == "__main__":

    vertex = Vertex(2,'d')
    print(vertex)

    # vertex.cells_idx[0] = 5
    # print(vertex.cells_idx)
    # print(vertex.Tcells_idx)
