#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
from enum import IntEnum
import numpy as np
import numpy.typing as npt

# Functions / Classes
class dir2D(IntEnum):
    """Returns a direction (dir) as an index. Used to help distinguish
       indices in an intuitive way when calling upon classes where one
       index represents direction. Only used for 2D."""

    N: int = 0 # North
    E: int = 1 # East
    S: int = 2 # South
    W: int = 3 # West

    # for primal faces
    Lp: int = 0 # Left - West
    Rp: int = 1 # Right - East
    Bp: int = 0 # Bottom - South
    Tp: int = 1 # Top - North

    # for dual faces
    Ld: int = 0 # Left - West
    Rd: int = 1 # Right - East
    Bd: int = 1 # Bottom - South
    Td: int = 0 # Top - North

    x:  int = 0 # x-coordinate
    y:  int = 1 # y-coordinate

    SW: int = 0 # South-West
    SE: int = 1 # South-East
    NE: int = 2 # North-East
    NW: int = 3 # North-West

def cosine_spacing(N: int, L: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    #### ============== ####
    #### Variable Setup ####
    #### ============== ####

    xp: npt.NDArray[np.float64]  = np.zeros((N+1), dtype = np.float64)   # grid points on primal grid
    xd: npt.NDArray[np.float64]  = np.zeros((N+2), dtype = np.float64)   # grid points on dual grid
    hxp: npt.NDArray[np.float64] = np.zeros((N),   dtype = np.float64)   # mesh width primal grid
    hxd: npt.NDArray[np.float64] = np.zeros((N+1), dtype = np.float64)   # mesh width dual grid

    #### ========= ####
    #### Execution ####
    #### ========= ####

    xd[0], xd[N+1] = 0, 1
    for i in range(N+1):
        xi = i*L/N
        xp[i] = 0.5*(1 - np.cos(np.pi*xi))       # x mesh point for primal mesh
        if i > 0:
            hxp[i-1] = xp[i] - xp[i-1]           # hx mesh width on primal mesh
            xd[i] = 0.5*(xp[i-1] + xp[i])        # x mesh point for dual mesh
    for i in range(N+1):
        hxd[i] = xd[i+1] - xd[i]                 # hx mesh width on dual mesh

    return xp, xd, hxp, hxd
