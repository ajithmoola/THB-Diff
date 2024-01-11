import numpy as np
from numba import njit


def compute_active_cells_and_coefficients(cells, fns, knotvectors, degrees):
    num_levels = max(cells.keys())
    dim = len(degrees)

    for lev in range(num_levels):
        curr_cells = cells[lev]
        curr_ac_cells = np.nonzero(curr_cells)
        
        # for cell in curr_ac_cells: