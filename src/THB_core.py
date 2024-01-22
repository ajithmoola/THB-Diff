import numpy as np
from numba import njit
from itertools import product
from funcs import *
from copy import deepcopy
from tqdm import tqdm

def compute_active_cells_active_supp(cells, fns, degrees):
    """
    Outputs a dictionary of active cells and their ovelapping non-zero basis functions from all levels
    Datastructure of ac_cells = Dict[lev: Dict[cellIdx: list[tuple(lev, fnIdx)]]]
    """
    # tested: working!
    num_levels = max(cells.keys())
    dims = len(degrees)
    ac_cells = {}

    for lev in range(num_levels):
        curr_ac_cells = list(zip(*np.nonzero(cells[lev])))
        curr_lev_ac_cells_ac_supp = {}
        for cell in curr_ac_cells:
            curr_lev_ac_cells_ac_supp[cell] = _compute_cell_active_supp(cell, lev, fns, degrees)
        ac_cells[lev] = curr_lev_ac_cells_ac_supp
    return ac_cells

def compute_fn_projection_matrices(fns, coeffs, degrees):
    # its working ig
    # TODO: verify the logic
    fn_coeffs = {}
    temp_fn_coeffs = {}
    ndim = len(degrees)
    max_lev = max(fns.keys())
    temp_fn_coeffs = {}
    for lev in range(max_lev-1, -1, -1):
        # TODO: this may not work if max_lev is 0
        curr_coeff = coeffs[lev]
        curr_lev_fn_coeffs = {}
        temp_curr_lev_fn_coeffs = {}
        for fn in np.ndindex(fns[lev].shape):
            # Get indices of children basis functions
            children = get_children_fns(fn, coeffs, lev, ndim)
            ac_children = [child for child in children if fns[lev+1][child]==1]
            # Vector projecting children basis functions to current level
            curr_fn_coeff = deepcopy([curr_coeff[dim] for dim in range(ndim)])
            non_zero_bool_arr_before_trunc = [coeff!=0 for coeff in curr_fn_coeff]
            # Truncates the basis function if any child function is active
            for ac_child in ac_children:
                for dim in range(ndim):
                    curr_fn_coeff[dim][fn[dim], ac_child[dim]] = 0
            # TODO: verify the logic
            if lev<(max_lev-1):
                for dim in range(ndim):
                    children_coeffs = deepcopy(coeffs[lev+1][dim])
                    for child in children:
                        children_coeffs[child[dim]] = temp_fn_coeffs[lev+1][child][dim][child[dim]]
                    curr_fn_coeff[dim] = curr_fn_coeff[dim] @ children_coeffs
            
            temp_curr_lev_fn_coeffs[fn] = curr_fn_coeff
            curr_lev_fn_coeffs[fn] = [curr_fn_coeff[dim][fn[dim]] for dim in range(ndim)]
        temp_fn_coeffs[lev] = temp_curr_lev_fn_coeffs
        fn_coeffs[lev] = curr_lev_fn_coeffs
    return fn_coeffs

def compute_active_span(params, knotvectors, cells, degrees, fn_shapes):
    max_lev = max(cells.keys())
    ndim = len(degrees)
    active_spans = []
    for param in params:
        for lev in range(max_lev, -1, -1):
            cellIdx = tuple(findSpan(fn_shapes[lev][dim], degrees[dim], param[dim], knotvectors[lev][dim])-degrees[dim] for dim in range(ndim))
            if cells[lev][cellIdx]==1:
                active_spans.append([lev, cellIdx])
                break
    return active_spans

def compute_tensor_product(params, ac_spans, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees):
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    PHI = []
    support_fns = []
    num_supp_cumsum = [0]
    for i, g in enumerate(tqdm(params)):
        cell_lev, cellIdx = ac_spans[i]
        cell_supp = ac_cells_supp[cell_lev][cellIdx]
        max_lev_cellIdx = [findSpan(fn_shapes[max_lev][dim], degrees[dim], g[dim], knotvectors[max_lev][dim]) - degrees[dim] for dim in range(ndim)]
        basis_fns = [basisFun(max_lev_cellIdx[dim]+degrees[dim], g[dim], degrees[dim], knotvectors[max_lev][dim]) for dim in range(ndim)]
        all_fn_values = []
        num_supp_cumsum.append(num_supp_cumsum[i]+len(cell_supp))
        for fn in cell_supp:
            fn_lev, fnIdx = fn
            sub_coeff = [fn_coeffs[fn_lev][fnIdx][dim][max_lev_cellIdx[dim]:max_lev_cellIdx[dim]+degrees[dim]+1] for dim in range(ndim)]
            sub_coeff_tp = tensor_product(sub_coeff)
            fn_tp = tensor_product(basis_fns)
            fn_value = np.sum(sub_coeff_tp*fn_tp)
            all_fn_values.append(fn_value)
        PHI.append(np.array(all_fn_values))

    return PHI, num_supp_cumsum

def PHIXCP(PHI, CP):
    pass
            
def get_children_fns(fnIdx, Coeff, level, dims):
    children = []
    
    for dim in range(dims):
        curr_coeff = Coeff[level][dim]
        children.append(np.nonzero(curr_coeff[fnIdx[dim]])[0])

    grids = np.meshgrid(*children)
    combinations = np.stack(grids, axis=-1).reshape(-1, dims)

    return [tuple(row) for row in combinations]
            
def get_supp_fns(cellIdx, degrees):
    ranges = [range(idx, idx+p+1) for idx, p in zip(cellIdx, degrees)]
    return [basisIdx for basisIdx in product(*ranges)]

def _compute_cell_active_supp(cellIdx, curr_level, fns, degrees):
    # tested: working!
    ac_supp = []
    for lev in range(curr_level, -1, -1):
        supp = get_supp_fns(cellIdx, degrees)
        for fn in supp:
            if fns[lev][fn]==1:
                ac_supp.append((lev, fn))
        #computing the parent cell index
        cellIdx = tuple(np.array(cellIdx)//2)
    return ac_supp

def compute_subdivision_coefficients(knotvectors, degrees):
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    subdivision_coeffs = {}
    for lev in range(max_lev):
        curr_coeff = {}
        for dim in range(ndim):
            knotvector = knotvectors[lev][dim]
            refined_knotvector = knotvectors[lev+1][dim]
            curr_coeff[dim] = assemble_Tmatrix(knotvector, refined_knotvector, knotvector.size, refined_knotvector.size, degrees[dim]).T
        subdivision_coeffs[lev] = curr_coeff
    return subdivision_coeffs

def compute_projection_to_highest_level(sub_coeffs):
    projected_sub_coeffs = {}
    max_lev = max(sub_coeffs.keys()) + 1
    ndim = len(sub_coeffs[0].keys())
    projected_sub_coeffs[max_lev-1] = sub_coeffs[max_lev-1]
    for lev in range(max_lev-2, -1, -1):
        curr_coeffs = {}
        for dim in range(ndim):
            curr_coeffs[dim] = sub_coeffs[lev][dim] @ projected_sub_coeffs[lev+1][dim]
        projected_sub_coeffs[lev] = curr_coeffs
    return projected_sub_coeffs