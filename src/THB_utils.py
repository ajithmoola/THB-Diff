import numpy as np
from numba import njit
from itertools import product
from funcs import tensor_product
from copy import deepcopy

def compute_active_cell_active_supp_and_coeff(cells, fns, degrees):
    """
    Outputs a dictionary of active cells and their ovelapping non-zero basis functions from all levels
    Datastructure of ac_cells = Dict[lev: Dict[cellIdx: list[tuple(lev, fnIdx)]]]
    """
    num_levels = max(cells.keys())
    dims = len(degrees)
    ac_cells = {}

    for lev in range(num_levels):
        curr_cells = cells[lev]
        curr_ac_cells = np.nonzero(curr_cells)
        curr_lev_ac_cells_ac_supp = {}
        for cell in curr_ac_cells:
            curr_lev_ac_cells_ac_supp[cell] = compute_cell_active_supp(cell, lev, fns, degrees)
        ac_cells[lev] = curr_lev_ac_cells_ac_supp
    return ac_cells

def compute_cell_active_supp(cellIdx, curr_level, fns, degrees):
    ac_supp = []
    for lev in range(curr_level, -1, -1):
        supp = get_supp_fns(cellIdx, degrees)
        for fn in supp:
            if fns[lev][fn]==1:
                ac_supp.append(lev, fn)
        #computing the parent cell index
        cellIdx = tuple(np.array(cellIdx)//2)
    return ac_supp

def compute_projection_matrices(fns, coeffs, degrees):
    # TODO: verify the logic
    fn_coeffs = {}
    dims = len(degrees)
    max_lev = max(fn.keys())
    for lev in range(max_lev-1, -1, -1):
        # TODO: this may not work if max_lev is 0
        curr_coeff = coeffs[lev]
        for fn in np.ndindex(fn[lev].shape):
            # Get indices of children basis functions 
            children = get_children_fns(fn, coeffs, lev, dims)
            ac_children = [child for child in children if fns[lev+1][child]==1]
            # Vector projecting children basis functions to current level
            curr_fn_coeff = deepcopy([curr_coeff[dim][fn[dim]] for dim in range(dims)])
            non_zero_ind_before_trunc = [coeff!=0 for coeff in curr_fn_coeff]
            # Truncates the basis function if any child function is active
            for ac_child in ac_children:
                for dim in range(dims):
                    curr_fn_coeff[dim][ac_child[dim]] = 0
            # Getting nonzero values except truncated children coefficient
            non_zero_fn_coeff = [curr_fn_coeff[dim][non_zero_ind_before_trunc[dim]!=0] for dim in range(dims)]
            
            # projects the subdivision coefficients from l -> l+1 to l -> max_lev
            if lev<(max_lev-1):
                for dim in range(dims):
                    projection_mat = np.zeros((4, 4))
                    for i, child in enumerate(children):
                        child_coeff = fn_coeffs[lev+1][child[dim]]
                        projection_mat[i] = child_coeff
                    non_zero_fn_coeff[dim] = non_zero_fn_coeff[dim].reshape(1, -1) @ projection_mat.T
            
            fn_coeffs[lev][fn] = non_zero_fn_coeff
    
    return fn_coeffs

def get_children_fns(fnIdx, Coeff, level, dims):
    children = []
    for dim in range(dims):
        curr_coeff = Coeff[level][dim]
        children.append(np.nonzero(curr_coeff[fnIdx[dim]])[0])

    grids = np.meshgrid(*children)
    combinations = np.stack(grids, axis=-1).reshape(-1, dims)

    return [tuple[row] for row in combinations]
            
def get_supp_fns(cellIdx, degrees):
    ranges = [range(idx, idx+p+1) for idx, p in zip(cellIdx, degrees)]
    return [basisIdx for basisIdx in product(*ranges)]

if __name__=="__main__":
    fns = get_supp_fns((0, 0, 0), (2, 2, 3))
    print(len(fns))