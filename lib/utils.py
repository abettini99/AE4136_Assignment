#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Library imports
from enum import IntEnum
import scipy.sparse as sparse
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

    V: int = 0 # Vertical
    H: int = 1 # Horizontal

    SW: int = 0 # South-West
    SE: int = 1 # South-East
    NE: int = 2 # North-East
    NW: int = 3 # North-West

def remove_sparse_rowcol(spmat: sparse.csr_matrix,
                         rows_idx: npt.NDArray[np.int32] = np.asarray(-1, dtype=np.int32),
                         cols_idx: npt.NDArray[np.int32] = np.asarray(-1, dtype=np.int32)) -> sparse.csr_matrix:
    """Remove rows and columns from an input sparse matrix. First removes rows,
       then removes columns.

       Parameters
       ----------
       spmat : sparse.csr_matrix
            Sparse matrix in compress sparse row format.

       rows_idx : array_like
            Rows to be truncated. The row numbering must be greater than zero
            for truncation to work.

       cols_idx : array_like
            Columns to be truncated. The column numbering must be greater
            than zero for truncation to work.

       Returns
       -------
       spmat : sparse.csr_matrix
            Truncated sparse matrix in compress sparse row format.

       Notes
       -----
       Be very careful using this function as no error message is thrown if
       either rows_idx or cols_idx has negative indices! It assumes that the
       input row and column indices are done correctly.
       # TODO: Add catch

       Examples
       --------
       rows_idx = np.asarray([0,3,4,5])
       spmat    = remove_sparse_rowcol(spmat, rows_idx=rows_idx))
       """

    if not isinstance(spmat, sparse.csr_matrix):
        raise TypeError("Sparse matrix not in csr form!")

    rows_mask: npt.NDArray[np.bool_] = np.ones(spmat.shape[0], dtype=bool)
    cols_mask: npt.NDArray[np.bool_] = np.ones(spmat.shape[1], dtype=bool)
    if rows_idx.min() >= 0:
        rows_mask[rows_idx] = False
    if cols_idx.min() >= 0:
        cols_mask[cols_idx] = False

    return spmat[rows_mask][:,cols_mask]
