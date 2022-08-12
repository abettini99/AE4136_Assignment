#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
from dataclasses import dataclass, field
# from copy import deepcopy

# Functions / Classes
@dataclass(order=True) # Not super necessary to have this decorator for this to work, but it is still nice to have.
class Face:

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
    vertices_idx:   list[int]   = field(init=False, repr=False)

    # truncated indices -> indices used when constructing with truncated solution set, e.g. no boundaries
    Tidx:           int         = field(init=False, repr=False)
    # Tcells_idx:     list[int]   = field(init=False, repr=False)
    # Tvertices_idx:  list[int]   = field(init=False, repr=False)

    # mirror indices -> index that face intersects with on the mirror mesh (i.e. dual edge intersection with primal edge)
    Midx:           int         = field(init=False, repr=False)

    def __post_init__(self) -> None:

        "Default configuration for faces, can be overwritten"

        self.idx            = -1
        self.sort_index     = self.idx

        if self.dim == 1:
            self.cells_idx      = [-1,-1]
            self.vertices_idx   = [None] # type: ignore
        elif self.dim == 2:
            self.cells_idx      = [-1,-1]
            self.vertices_idx   = [-1,-1]
        elif self.dim == 3:
            self.cells_idx      = [-1,-1]
            self.vertices_idx   = [-1,-1,-1,-1]
        else:
            raise ValueError(f'Face created with incorrect dimensions')

        self.Tidx               = -1
        self.Midx               = -1
        # self.Tcells_idx         = deepcopy(self.cells_idx)
        # self.Tvertices_idx      = deepcopy(self.vertices_idx)

    def set_idx(self, idx: int) -> None:
        self.idx = idx
        self.sort_index = self.idx

    def set_Tidx(self, Tidx: int) -> None:
        self.Tidx = Tidx

    def __str__(self) -> str:
        return  f'{self.dim}D Face: {self.type}\n'+ \
                f'idx, Tidx: {self.idx}, {self.Tidx}\n'+ \
                f'cell neighbours = {self.cells_idx}\n'+ \
                f'vertex neighbours = {self.vertices_idx}\n'

if __name__ == "__main__":

    face = Face(2,'b')
    print(face)
