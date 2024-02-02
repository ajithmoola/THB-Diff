import numpy as np
from itertools import product
from funcs import compute_coeff_tensor_product, compute_projection, compute_tensor_product, findSpan, basisFun, assemble_Tmatrix
from copy import deepcopy
from tqdm import tqdm
import torch
import THB_eval


class THBEval(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum, device):
        
        ctx.save_for_backward(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        ctx.device = device
        if device=='cuda':
            return THB_eval.forward(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        else:
            return THB_eval.cpp_forward(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)

    @staticmethod
    def backward(ctx, grad_output):
        ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum = ctx.saved_tensors
        device = ctx.device
        if device=='cuda':
            grad_ctrl_pts = THB_eval.backward(grad_output, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        else:
            grad_ctrl_pts = THB_eval.cpp_backward(grad_output, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        return grad_ctrl_pts, None, None, None, None

def compute_active_cells_active_supp(cells, fns, degrees):
    max_lev = max(cells.keys())
    ac_cells = {}

    for lev in range(max_lev+1):
        curr_ac_cells = list(zip(*np.nonzero(cells[lev])))
        curr_lev_ac_cells_ac_supp = {}
        for cell in curr_ac_cells:
            curr_lev_ac_cells_ac_supp[cell] = _compute_cell_active_supp(cell, lev, fns, degrees)
        ac_cells[lev] = curr_lev_ac_cells_ac_supp
    return ac_cells

def compute_fn_projection_matrices(fns, coeffs, degrees):
    max_lev = max(fns.keys())
    ndim = len(degrees)
    fn_coeffs = {lev:{} for lev in range(max_lev)}
    coeffs_tp = {lev: compute_coeff_tensor_product([coeffs[lev][dim] for dim in range(ndim)]) for lev in range(ndim)}
    
    projection_coeff = {}
    projection_coeff[max_lev-2] = coeffs_tp[max_lev-1]
    for lev in range(max_lev-3, -1, -1):
        projection_coeff[lev] = compute_projection([coeffs_tp[lev+1], projection_coeff[lev+1]])

    for lev in range(max_lev-1, -1, -1):
        next_lev_fns = fns[lev+1]
        curr_coeff_tp = deepcopy(coeffs_tp[lev])
        
        for fn in np.ndindex(fns[lev].shape):
            for child in np.ndindex(fns[lev+1].shape):
                if fns[lev+1][child]==1:
                    curr_coeff_tp[fn][child] = 0

        if lev<(max_lev-1):
            curr_coeff_tp = compute_projection([curr_coeff_tp, projection_coeff[lev]])
        
        fn_coeffs[lev] = curr_coeff_tp

    return fn_coeffs

def compute_active_span(params, knotvectors, cells, degrees, fn_shapes):
    max_lev = max(cells.keys())
    ndim = len(degrees)
    active_spans = []
    for param in params:
        for lev in range(max_lev, -1, -1):
            cellIdx = tuple(findSpan(fn_shapes[lev][dim]-1, degrees[dim], param[dim], knotvectors[lev][dim])-degrees[dim] for dim in range(ndim))
            if cells[lev][cellIdx]==1:
                active_spans.append((lev, cellIdx))
                break
    return active_spans

def compute_tensor_product(params, ac_spans, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees):
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    PHI = []
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
            slice_tuple = tuple(slice(max_lev_cellIdx[dim]-degrees[dim], max_lev_cellIdx[dim]+1) for dim in range(ndim))
            sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
            fn_tp = compute_tensor_product(basis_fns)
            fn_value = np.sum(sub_coeff*fn_tp)
            all_fn_values.append(fn_value)
        PHI.append(np.array(all_fn_values))

    return PHI, num_supp_cumsum

def evaluate(PHI, ctrl_pts, ac_spans, num_supp_cumsum, ac_cells_ac_supp):
    output = np.zeros((len(PHI), ctrl_pts[0].shape[-1]))
    for i, ac_cell in enumerate(ac_spans):
        supp = ac_cells_ac_supp[ac_cell[0]][ac_cell[1]]
        phi = PHI[i]
        for j, (lev, fn) in enumerate(supp):
            output[i, :] += phi[j]*ctrl_pts[lev][fn]
    return output

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
    ac_supp = []
    for lev in range(curr_level, -1, -1):
        supp = get_supp_fns(cellIdx, degrees)
        for fn in supp:
            if fns[lev][fn]==1:
                ac_supp.append((lev, fn))
        cellIdx = tuple(np.array(cellIdx)//2)
    return ac_supp

def support_cells_multi(knot_vectors, degrees, fn):
    all_support_cells = []

    for dim, (knot_vector, degree, i) in enumerate(zip(knot_vectors, degrees, fn)):
        start = max(i - degree, 0)
        end = min(i+1, len(np.unique(knot_vector)) - 1)
        cell_indices = set()
        for j in range(start, end):
            cell_indices.add(j)
        
        all_support_cells.append(sorted(cell_indices))

    support_cells_md = np.array(np.meshgrid(*all_support_cells, indexing='ij')).T.reshape(-1, len(degrees))

    return [tuple(cell) for cell in support_cells_md]

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