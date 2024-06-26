import numpy as np
from itertools import product
from copy import deepcopy
import jax.numpy as jnp
from jax import jacfwd, jit, lax
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, List

from THB.bspline_funcs import *
from THB.utils import timer


def compute_active_cells_active_supp(
    cells: Dict[int, np.ndarray], fns: Dict[int, np.ndarray], degrees: Tuple[int]
) -> Dict[int, Dict[Tuple[int], List[Tuple[int]]]]:
    """
    Compute the active cells and their active supports.

    Args:
        cells (dict): Dictionary containing the cells at each level.
        fns (dict): Dictionary containing the basis functions at each level.
        degrees (list): List of degrees for each dimension.

    Returns:
        dict: Dictionary containing the active cells and their active
        supports at each level.
    """
    max_lev = max(cells.keys())
    ac_cells = {}

    for lev in range(max_lev + 1):
        curr_ac_cells = list(zip(*np.nonzero(cells[lev])))
        curr_lev_ac_cells_ac_supp = {}
        for cell in curr_ac_cells:
            curr_lev_ac_cells_ac_supp[cell] = _compute_cell_active_supp(
                cell, lev, fns, degrees
            )
        ac_cells[lev] = curr_lev_ac_cells_ac_supp
    return ac_cells


def compute_refinement_operators(
    fns: Dict[int, np.ndarray],
    coeffs: Dict[int, Dict[int, np.ndarray]],
    degrees: Tuple[int],
) -> Dict[int, np.ndarray]:
    """
    Compute the projection matrices for the basis functions.

    Args:
        fns (dict): Dictionary containing the status of basis functions
        at each level.
        coeffs (dict): Dictionary containing the subdivision coefficients
        at each level for each dimension.
        degrees (tuple): tuple of degrees for each dimension.

    Returns:
        dict: Dictionary containing the refinement operators for the
        basis functions at each level.
    """
    max_lev = max(fns.keys())
    ndim = len(degrees)
    fn_coeffs = {lev: {} for lev in range(max_lev + 1)}

    # Computes the tensor product of the projection coefficients from i-th level to (i+1)-th level
    # For 2D tensorproduct THB-splines the transformation can be implemented as an einsum
    # (ij, kl) -> ikjl
    # For 3D tensorproduct THB-splines
    # (ij, kl, mn) -> ikmjln

    def compute_coeff_tensor_product(args):
        if len(args) == 2:
            return np.einsum("ij, kl -> ikjl", *args, optimize=True)
        elif len(args) == 3:
            return np.einsum("ij, kl, mn -> ikmjln", *args, optimize=True)

    coeffs_tp = {
        lev: compute_coeff_tensor_product([coeffs[lev][dim] for dim in range(ndim)])
        for lev in range(max_lev)
    }

    projection_coeff = {}
    projection_coeff[max_lev - 2] = coeffs_tp[max_lev - 1]

    def compute_projection(args, ndim):
        if ndim == 2:
            return np.einsum("ijkl, klmn -> ijmn", *args, optimize=True)
        elif ndim == 3:
            return np.einsum("ijklmn, lmnopq -> ijkopq", *args, optimize=True)

    # Computes the projection coefficients for all levels to the finest level
    for lev in range(max_lev - 3, -1, -1):
        projection_coeff[lev] = compute_projection(
            [coeffs_tp[lev + 1], projection_coeff[lev + 1]], ndim=ndim
        )

    # Truncation of basis functions
    for lev in range(max_lev - 1, -1, -1):
        curr_coeff_tp = deepcopy(coeffs_tp[lev])
        trunc_indices = np.where(fns[lev + 1] == 1)

        for fn in np.ndindex(fns[lev].shape):
            curr_coeff_tp[fn][trunc_indices] = 0

        if lev < (max_lev - 1):
            curr_coeff_tp = compute_projection(
                [curr_coeff_tp, projection_coeff[lev]], ndim=ndim
            )

        fn_coeffs[lev] = jnp.array(curr_coeff_tp).astype(jnp.float16)

    fn_coeffs[max_lev] = jnp.ones((*fns[max_lev].shape, *fns[max_lev].shape))
    return fn_coeffs


def compute_active_span_v2(
    params, knotvectors, cells, degrees, ac_cells_ac_supp, fn_coeffs
):
    max_lev = max(cells.keys())
    ndim = len(degrees)
    spans = {}
    active_spans = []
    num_supp = []
    sub_coeffs = []

    for lev in range(max_lev + 1):
        curr_spans = np.array(
            [
                np.array(
                    find_span_array_jax(
                        params[:, dim], knotvectors[lev][dim], degrees[dim]
                    )
                )
                - degrees[dim]
                for dim in range(ndim)
            ]
        ).T
        spans[lev] = tuple(tuple(row) for row in curr_spans)

    for i in tqdm(range(len(params))):
        for lev in range(max_lev, -1, -1):
            cellIdx = spans[lev][i]

            if cells[lev][cellIdx] == 1:
                active_spans.append((lev, cellIdx))
                curr_supp_fns = ac_cells_ac_supp[lev][cellIdx]
                num_supp.append(len(curr_supp_fns))
                sub_coeffs += [
                    fn_coeffs[fn_lev][fnIdx] for fn_lev, fnIdx in curr_supp_fns
                ]
                break

    return active_spans, jnp.array(num_supp), sub_coeffs


def compute_active_span(params, knotvectors, cells, degrees, ac_cells_ac_supp):
    max_lev = max(cells.keys())
    ndim = len(degrees)
    spans = {}
    active_spans = []
    active_cell_supp = []
    num_supp = []

    for lev in range(max_lev + 1):
        curr_spans = np.array(
            [
                np.array(
                    find_span_array_jax(
                        params[:, dim], knotvectors[lev][dim], degrees[dim]
                    )
                )
                - degrees[dim]
                for dim in range(ndim)
            ]
        ).T
        spans[lev] = tuple(tuple(row) for row in curr_spans)

    for i in tqdm(range(len(params))):
        for lev in range(max_lev, -1, -1):
            cellIdx = spans[lev][i]

            if cells[lev][cellIdx] == 1:
                cell_supp = ac_cells_ac_supp[lev][cellIdx]
                active_cell_supp.append(cell_supp)
                num_supp.append(len(cell_supp))
                break

    return active_cell_supp, num_supp


def compute_THB_fns_diff(
    params, ac_spans, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees
):
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)

    for i, g in enumerate(tqdm(params)):
        cell_lev, cellIdx = ac_spans[i]
        cell_supp = ac_cells_supp[cell_lev][cellIdx]


def compute_THB_fns_tp(
    params: np.ndarray,
    ac_spans: List[Tuple[int]],
    ac_cells_supp: Dict[int, Dict[Tuple[int], List[Tuple[int]]]],
    fn_coeffs: Dict[int, np.ndarray],
    fn_shapes: Dict[int, Tuple[int]],
    knotvectors: Dict[int, Dict[int, np.ndarray]],
    degrees: Tuple[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the basis function tensor product for given parameters.

    Args:
        params (list): List of parameter values.
        ac_spans (list): List of active spans for the given parameters.
        ac_cells_supp (dict): Dictionary containing the active cells and their
        supports at each level.
        fn_coeffs (dict): Dictionary containing the refinement operator of
        basis functions at each level.
        fn_shapes (dict): Dictionary containing the shape of basis functions at
        each level.
        knotvectors (dict): Dictionary containing the knot vectors at each level.
        degrees (tuple): Tuple of degrees for each dimension.

    Returns:
        numpy.ndarray: Flattened array of basis function values.
        numpy.ndarray: Array of cumulative sums of the number of supports.
        numpy.ndarray: Array of number of supports for each parameter.
    """
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    PHI = []
    num_supp = []

    for i, g in enumerate(tqdm(params)):
        cell_lev, cellIdx = ac_spans[i]
        cell_supp = ac_cells_supp[cell_lev][cellIdx]

        max_lev_cellIdx = [
            findSpan(
                fn_shapes[max_lev][dim], degrees[dim], g[dim], knotvectors[max_lev][dim]
            )
            for dim in range(ndim)
        ]

        basis_fns = [
            basisFun(
                max_lev_cellIdx[dim], g[dim], degrees[dim], knotvectors[max_lev][dim]
            )
            for dim in range(ndim)
        ]

        num_supp.append(len(cell_supp))

        slice_tuple = tuple(
            slice(max_lev_cellIdx[dim] - degrees[dim], max_lev_cellIdx[dim] + 1)
            for dim in range(ndim)
        )
        fn_tp = compute_tensor_product(basis_fns)
        for fn in cell_supp:
            fn_lev, fnIdx = fn
            sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
            fn_value = np.sum(sub_coeff * fn_tp)
            PHI.append(fn_value)

    return np.array(PHI), np.array(num_supp)


def CP_arr_to_dict(CP_arr, sh_fns, num_levels):
    CP_arr = np.array(CP_arr)
    nCP = np.array([0] + [np.prod(sh_fns[lev]) for lev in range(num_levels)]).cumsum()
    ctrl_pts = {
        lev: CP_arr[nCP[lev] : nCP[lev + 1]].reshape(*sh_fns[lev], CP_arr.shape[1])
        for lev in range(num_levels)
    }
    return ctrl_pts


def refine_ctrl_pts(CP, Coeffs, curr_fn_state, prev_fn_state, sh_fns, num_levels, ndim):
    for lev in range(1, num_levels):
        curr_coeff = Coeffs[lev]
        for fn in np.ndindex(sh_fns[lev]):
            if prev_fn_state[lev][fn] == 0 and curr_fn_state[lev][fn] == 1:
                refine_coeff = [curr_coeff[dim].T[fn[dim]] for dim in range(ndim)]
                refine_coeff_tp = compute_tensor_product(refine_coeff)
                CP[lev][CP] = np.sum(
                    refine_coeff_tp[..., np.newaxis] * CP[lev - 1],
                    axis=tuple(range(len(refine_coeff_tp.shape))),
                )
    return CP


def compute_multilevel_bezier_extraction_operators(
    params: List[List[float]],
    ac_spans: List[Tuple[int]],
    ac_cells_supp: Dict[int, Dict[Tuple[int], List[Tuple[int]]]],
    fn_coeffs: Dict[int, Dict[int, np.ndarray]],
    cell_shapes: Dict[int, Tuple[int]],
    fn_shapes: Dict[int, Tuple[int]],
    knotvectors: Dict[int, Dict[int, np.ndarray]],
    degrees: Tuple[int],
) -> List[np.ndarray]:
    """
    Compute the multilevel Bezier extraction operators.

    Args:
        params (list): List of parameter values.
        ac_spans (list): List of active spans.
        ac_cells_supp (dict): Dictionary containing the active cells and their
        supports at each level.
        fn_coeffs (dict): Dictionary containing the coefficients of basis
        functions at each level.
        fn_shapes (dict): Dictionary containing the shape of basis functions at
        each level.
        knotvectors (dict): Dictionary containing the knot vectors at each level.
        degrees (list): List of degrees for each dimension.

    Returns:
        (list): List of beizer extraction operators
    """
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)

    Cmax = [
        bezier_extraction(knotvectors[max_lev][dim], degrees[dim])
        for dim in range(ndim)
    ]

    Cmax_tp = np.zeros(
        (
            *cell_shapes[max_lev],
            *tuple(np.array(degrees) + 1),
            *tuple(np.array(degrees) + 1),
        )
    )

    def compute_coeff_tensor_product(args):
        if len(args) == 2:
            return np.einsum("ij, kl -> ikjl", *args, optimize=True)
        elif len(args) == 3:
            return np.einsum("ij, kl, mn -> ikmjln", *args, optimize=True)

    for cell in np.ndindex(cell_shapes[max_lev]):
        Cmax_tp[cell] = compute_coeff_tensor_product(
            [Cmax[dim][cell[dim]] for dim in range(ndim)]
        )

    def compute_bezier_projection(args, ndim):
        if ndim == 2:
            return np.einsum("ij, ijkl -> kl", *args, optimize=True)
        elif ndim == 3:
            return np.einsum("ijk, ijklmn -> lmn", *args, optimize=True)

    C = []
    for i, g in enumerate(tqdm(params)):
        cell_lev, cellIdx = ac_spans[i]
        ac_supp = ac_cells_supp[cell_lev][cellIdx]
        max_lev_cellIdx = [
            findSpan(
                fn_shapes[max_lev][dim], degrees[dim], g[dim], knotvectors[max_lev][dim]
            )
            for dim in range(ndim)
        ]
        Cmax_local = Cmax_tp[
            tuple(max_lev_cellIdx[dim] - degrees[dim] for dim in range(ndim))
        ]
        C_local = np.zeros((len(ac_supp), *tuple(np.array(degrees) + 1)))
        for i, fn in enumerate(ac_supp):
            fn_lev, fnIdx = fn
            slice_tuple = tuple(
                slice(max_lev_cellIdx[dim] - degrees[dim], max_lev_cellIdx[dim] + 1)
                for dim in range(ndim)
            )
            sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
            C_local[i] = compute_bezier_projection([sub_coeff, Cmax_local], ndim=ndim)
        C.append(C_local)
    return C


def get_children_fns(
    fnIdx: Tuple[int], Coeff: Dict[int, Dict[int, np.ndarray]], level: int, dims: int
) -> List[Tuple[int]]:
    """
    Get the children basis functions for a given basis function.

    Args:
        fnIdx (tuple): Index of the basis function.
        Coeff (dict): Dictionary containing the coefficients of basis functions at each level.
        level (int): Level of the basis function.
        dims (int): Number of dimensions.

    Returns:
        list: List of children basis functions.
    """
    children = []

    for dim in range(dims):
        curr_coeff = Coeff[level][dim]
        children.append(np.nonzero(curr_coeff[fnIdx[dim]])[0])

    grids = np.meshgrid(*children)
    combinations = np.stack(grids, axis=-1).reshape(-1, dims)

    return [tuple(row) for row in combinations]


def get_supp_fns(cellIdx: Tuple[int], degrees: Tuple[int]) -> List[Tuple[int]]:
    """
    Get the support basis functions for a given cell.

    Args:
        cellIdx (tuple): Index of the cell.
        degrees (tuple): Tuple of degrees for each dimension.

    Returns:
        list: List of support basis functions.
    """
    ranges = [range(idx, idx + p + 1) for idx, p in zip(cellIdx, degrees)]
    return [basisIdx for basisIdx in product(*ranges)]


def _compute_cell_active_supp(
    cellIdx: Tuple[int],
    curr_level: int,
    fns: Dict[int, np.ndarray],
    degrees: Tuple[int],
) -> List[Tuple[int]]:
    """
    Compute the active supports for a given cell.

    Args:
        cellIdx (tuple): Index of the cell.
        curr_level (int): Current level.
        fns (dict): Dictionary containing the basis functions at each level.
        degrees (list): List of degrees for each dimension.

    Returns:
        list: List of active supports for the given cell.
    """
    ac_supp = []
    for lev in range(curr_level, -1, -1):
        supp = get_supp_fns(cellIdx, degrees)
        for fn in supp:
            if fns[lev][fn] == 1:
                ac_supp.append((lev, fn))
        cellIdx = tuple(np.array(cellIdx) // 2)
    return ac_supp


def support_cells_multi(
    knot_vectors: Dict[int, np.ndarray], degrees: Tuple[int], fn: Tuple[int]
) -> List[Tuple[int]]:
    """
    Compute the support cells for a given function.

    Args:
        knot_vectors (Dict): Dictionary of knot vectors for each dimension.
        degrees (tuple): Tuple of degrees for each dimension.
        fn (tuple): Tuple of indices for each dimension.

    Returns:
        list: List of support cells for the given function.
    """
    all_support_cells = []

    for dim, (knot_vector, degree, i) in enumerate(zip(knot_vectors, degrees, fn)):
        start = max(i - degree, 0)
        end = min(i + 1, len(np.unique(knot_vector)) - 1)
        cell_indices = set()
        for j in range(start, end):
            cell_indices.add(j)

        all_support_cells.append(sorted(cell_indices))

    support_cells_md = np.array(
        np.meshgrid(*all_support_cells, indexing="ij")
    ).T.reshape(-1, len(degrees))

    return [tuple(cell) for cell in support_cells_md]


def compute_subdivision_coefficients(
    knotvectors: Dict[int, Dict[int, np.ndarray]], degrees: Tuple[int]
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Compute the subdivision coefficients.

    Args:
        knotvectors (dict): Dictionary containing the knot vectors at each level.
        degrees (tuple): Tuple of degrees for each dimension.

    Returns:
        dict: Dictionary containing the subdivision coefficients.
    """
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    subdivision_coeffs = {}
    for lev in range(max_lev):
        curr_coeff = {}
        for dim in range(ndim):
            knotvector = knotvectors[lev][dim]
            refined_knotvector = knotvectors[lev + 1][dim]
            curr_coeff[dim] = assemble_Tmatrix(
                knotvector,
                refined_knotvector,
                knotvector.size,
                refined_knotvector.size,
                degrees[dim],
            ).T
        subdivision_coeffs[lev] = curr_coeff
    return subdivision_coeffs


######### Parallel Basis Function Tensorproduct Computation ###########


def basis_fn_worker(
    param_idx,
    param,
    ac_cells_supp,
    fn_coeffs,
    fn_shapes,
    knotvectors,
    degrees,
):
    """
    Worker function for parallel computation of basis function tensor product.

    Args:
        param_idx (int): Index of the parameter.
        param (tuple): Parameter values.
        ac_spans (list): List of active spans.
        ac_cells_supp (dict): Dictionary containing the active cells and their supports at each level.
        fn_coeffs (dict): Dictionary containing the coefficients of basis functions at each level.
        fn_shapes (dict): Dictionary containing the shape of basis functions at each level.
        knotvectors (dict): Dictionary containing the knot vectors at each level.
        degrees (list): List of degrees for each dimension.

    Returns:
        tuple: Tuple containing the basis function values and the number of supports.
    """
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    cell_supp = ac_cells_supp[param_idx]

    max_lev_cellIdx = [
        findSpan(
            fn_shapes[max_lev][dim], degrees[dim], param[dim], knotvectors[max_lev][dim]
        )
        for dim in range(ndim)
    ]

    basis_fns = [
        basisFun_jax(param[dim], knotvectors[max_lev][dim], degrees[dim]).squeeze()
        for dim in range(ndim)
    ]
    slice_tuple = tuple(
        slice(max_lev_cellIdx[dim] - degrees[dim], max_lev_cellIdx[dim] + 1)
        for dim in range(ndim)
    )
    all_fn_values = []
    fn_tp = compute_tensor_product(basis_fns)
    for fn in cell_supp:
        fn_lev, fnIdx = fn
        sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
        fn_value = jnp.sum(sub_coeff * fn_tp)
        all_fn_values.append(fn_value)
    return all_fn_values


def THB_basis_fns_tp_parallel(
    params, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees
):
    """
    Compute the basis function tensor product in parallel.

    Args:
        params (ndarray): List of parameter values.
        ac_spans (list): List of active spans.
        ac_cells_supp (dict): Dictionary containing the active cells and their supports at each level.
        fn_coeffs (dict): Dictionary containing the coefficients of basis functions at each level.
        fn_shapes (dict): Dictionary containing the shape of basis functions at each level.
        knotvectors (dict): Dictionary containing the knot vectors at each level.
        degrees (list): List of degrees for each dimension.

    Returns:
        numpy.ndarray: Array of basis function values.
        numpy.ndarray: Array of number of supports for each parameter.
    """
    # fn_coeffs = jnp.array(fn_coeffs)
    with Pool(processes=cpu_count()) as pool:
        tasks = [
            (i, g, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees)
            for i, g in enumerate(params)
        ]

        PHI = pool.starmap(basis_fn_worker, tasks)

    return jnp.array(PHI)


def THB_basis_fns(
    params: jnp.ndarray,
    ac_cells_supp: List[List[int, Tuple[int]]],
    num_supp: List[int],
    fn_coeffs: Dict[int : np.ndarray],
    fn_shapes: Dict[int, Tuple[int]],
    knotvectors: Dict[int, Tuple[np.ndarray]],
    degrees: Tuple[int],
    return_mode: List[bool, bool] = [True, False],
) -> jnp.ndarray:
    """Computes THB-spline tensor product of basis functions or derivates of the basis functions

    Args:
        params (ndarray): an array of parametric coordinates
        ac_cells_supp (list): A list of tuples which contains indices of non-zero basis functions corresponding to the parameter value
        num_supp (list): a list containing number of non-zero basis functions for at each parameteric coordinate
        fn_coeffs (dict): a dictionary of ndarrays containing coefficients for each basis function in every level which projects them to the highest level
        fn_shapes (dict): a dictionary of tuples describing the number of the basis functions in the tensor product along parametric directions
        knotvectors (dict): a dictionary of tuples of ndarrays representing knotvectors
        degrees (tuple): degree of each dimension
        return_mode (list, optional): If [True, False] only the basis functions are returned, elif [False, True] only the derivatives of basis functions are returned, elif [True, True] both are returned. Defaults to [True, False].

    Returns:
        ndarray or tuple of ndarrays: flattened tensor product of basis functions or derivatives of basis functions or both
    """
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)

    n_fns = np.zeros(max_lev + 2, dtype=np.int_)
    for lev in range(1, max_lev + 2):
        n_fns[lev] = n_fns[lev - 1] + np.prod(fn_shapes[lev - 1])

    fn_coeffs_flat = [
        jnp.array(fn_coeffs[lev])
        .reshape(-1, *fn_coeffs[lev].shape[ndim:])
        .astype(jnp.float32)
        for lev in range(max_lev + 1)
    ]

    fn_coeffs = jnp.vstack(fn_coeffs_flat)
    Jm = jnp.array(
        [
            n_fns[fn_lev] + np.ravel_multi_index(fnIdx, fn_shapes[fn_lev])
            for cell_supp in ac_cells_supp
            for fn_lev, fnIdx in cell_supp
        ]
    )

    spans = [
        find_span_array_jax(params[:, dim], knotvectors[max_lev][dim], degrees[dim])
        - degrees[dim]
        for dim in range(ndim)
    ]

    fn_coeffs_per_param = fn_coeffs[Jm]

    def extract_slices(i):
        starts = [spans[dim][i] for dim in range(ndim)]
        sizes = [degrees[dim] + 1 for dim in range(ndim)]
        return lax.dynamic_slice(fn_coeffs_per_param[i], starts, sizes)

    # Vectorize the function across the appropriate axis
    vectorized_extraction = vmap(extract_slices, in_axes=0)

    # Apply the vectorized function
    fn_coeffs_indexed = vectorized_extraction(jnp.arange(fn_coeffs_per_param.shape[0]))

    def get_basis_fns(basis_funcs):
        if ndim == 3:
            basis_fn_tp = jnp.einsum("ij, ik, il -> ijkl", *basis_funcs)
        elif ndim == 2:
            basis_fns_tp = jnp.einsum("ij, ik -> ijk", *basis_funcs)
        repeat_ind = jnp.repeat(jnp.arange(basis_fns_tp.shape[0]), jnp.array(num_supp))
        PHI = jnp.einsum("i... -> i", basis_fns_tp[repeat_ind] * fn_coeffs_indexed)
        return PHI

    if return_mode[0] and not return_mode[1]:
        basis_fns = [
            basis_fns_vmap(params[:, dim], knotvectors[max_lev][dim], degrees[dim])
            for dim in range(ndim)
        ]
        return get_basis_fns(basis_fns)
    elif not return_mode[0] and return_mode[1]:
        der_basis_fns = [
            der_basis_fns_vmap(params[:, dim], knotvectors[max_lev][dim], degrees[dim])
            for dim in range(ndim)
        ]
        return get_basis_fns(der_basis_fns)
    elif return_mode[0] and return_mode[1]:
        basis_fns = [
            basis_fns_vmap(params[:, dim], knotvectors[max_lev][dim], degrees[dim])
            for dim in range(ndim)
        ]
        der_basis_fns = [
            der_basis_fns_vmap(params[:, dim], knotvectors[max_lev][dim], degrees[dim])
            for dim in range(ndim)
        ]
        return get_basis_fns(basis_fns), get_basis_fns(der_basis_fns)
