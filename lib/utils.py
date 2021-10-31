#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
from enum import IntEnum

# Functions / Classes
class dir2D(IntEnum):
    """Returns a direction (dir) as an index. Used to help distinguish
       indices in an intuitive way when calling upon classes where one
       index represents direction. Only used for 2D."""

    N: int = 0 # North
    E: int = 1 # East
    S: int = 2 # South
    W: int = 3 # West

    H: int = 0 # Horizontal
    V: int = 1 # Vertical
