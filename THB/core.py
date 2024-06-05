import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jit, lax
import jax
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, List
from functools import partial

from THB.bspline_funcs import *
from THB.jax_funcs import Evaluate_JAX


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

    return active_cell_supp, np.array(num_supp)


def THB_basis_fns_tp_serial(
    params, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees
):
    """
    Compute the basis function tensor product in a serial manner.

    Args:
        params (ndarray): List of parameter values.
        ac_cells_supp (dict): Dictionary containing the active cells and their supports at each level.
        fn_coeffs (dict): Dictionary containing the coefficients of basis functions at each level.
        fn_shapes (dict): Dictionary containing the shape of basis functions at each level.
        knotvectors (dict): Dictionary containing the knot vectors at each level.
        degrees (list): List of degrees for each dimension.

    Returns:
        numpy.ndarray: Array of basis function values.
        numpy.ndarray: Array of number of supports for each parameter.
    """
    params = np.array(params)

    def basis_fn_worker(param_idx, param):
        max_lev = max(knotvectors.keys())
        ndim = len(degrees)
        cell_supp = ac_cells_supp[param_idx]

        max_lev_cellIdx = [
            findSpan(
                fn_shapes[max_lev][dim],
                degrees[dim],
                param[dim],
                knotvectors[max_lev][dim],
            )
            for dim in range(ndim)
        ]

        basis_fns = [
            basisFun(
                max_lev_cellIdx[dim],
                param[dim],
                degrees[dim],
                knotvectors[max_lev][dim],
            )
            for dim in range(ndim)
        ]

        slice_tuple = tuple(
            slice(max_lev_cellIdx[dim] - degrees[dim], max_lev_cellIdx[dim] + 1)
            for dim in range(ndim)
        )
        all_fn_values = []
        all_fn_coeffs = []
        fn_tp = compute_tensor_product(basis_fns)
        for fn in cell_supp:
            fn_lev, fnIdx = fn
            sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
            fn_value = np.sum(sub_coeff * fn_tp)
            all_fn_values.append(fn_value)
            all_fn_coeffs.append(sub_coeff)
        return (
            all_fn_values,
            np.array(max_lev_cellIdx),
            np.array(all_fn_coeffs),
        )

    PHI = []
    spans = []
    COEFF = []
    for i, g in tqdm(enumerate(params)):
        a, b, c = basis_fn_worker(i, g)
        PHI.append(a)
        spans.append(b)
        COEFF.append(c)

    return np.array(PHI).reshape(-1)


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

    # basis_fns = [
    # basisFun_jax(param[dim], knotvectors[max_lev][dim], degrees[dim]).squeeze()
    # for dim in range(ndim)
    # ]

    basis_fns = [
        basisFun(
            max_lev_cellIdx[dim], param[dim], degrees[dim], knotvectors[max_lev][dim]
        )
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
        fn_value = np.sum(sub_coeff * fn_tp)
        all_fn_values.append(fn_value)

    return all_fn_values


def THB_basis_fns_tp_parallel(
    params, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees
):
    """
    Compute the basis function tensor product in parallel.

    Args:
        params (ndarray): List of parameter values.
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

    return np.array(PHI).reshape(-1)


####################### Testing ############################


@partial(jit, static_argnums=(4))
def THB_basis_fns(
    params: jnp.ndarray,
    fn_coeffs_indexed: jnp.ndarray,
    repeat_ind: List[int],
    max_knotvectors: Tuple[np.ndarray],
    degrees: Tuple[int],
) -> jnp.ndarray:
    ndim = len(degrees)

    basis_fns = [
        basis_fns_vmap(params[:, dim], max_knotvectors[dim], degrees[dim])
        for dim in range(ndim)
    ]

    if ndim == 3:
        basis_fns_tp = jnp.einsum("ij, ik, il -> ijkl", *basis_fns)
    elif ndim == 2:
        basis_fns_tp = jnp.einsum("ij, ik -> ijk", *basis_fns)

    PHI = jnp.einsum("i... -> i", basis_fns_tp[repeat_ind] * fn_coeffs_indexed)

    return PHI.reshape(-1, 1)


@partial(jit, static_argnums=(4))
def THB_basis_fns_Jacobian(
    params: jnp.ndarray,
    fn_coeffs_indexed: jnp.ndarray,
    repeat_ind: List[int],
    max_knotvectors: Tuple[np.ndarray],
    degrees: Tuple[int],
):
    jac = jacfwd(THB_basis_fns, argnums=0)(
        params, fn_coeffs_indexed, repeat_ind, max_knotvectors, degrees
    )
    indices = jnp.arange(params.shape[0])
    return jac[:, :, indices, :]


@partial(jit, static_argnums=(6))
def THB_evaluate(
    params, ctrl_pts, fn_coeffs_indexed, repeat_ind, Jm, knotvectors, degrees
):
    num_pts = params.shape[0]
    PHI = THB_basis_fns(params, fn_coeffs_indexed, repeat_ind, knotvectors, degrees)
    output = Evaluate_JAX(ctrl_pts, Jm, PHI, repeat_ind, num_pts)
    prod = PHI * ctrl_pts[Jm]
    output = jnp.zeros((num_pts, ctrl_pts.shape[1])).at[repeat_ind].add(prod)
    return output


@partial(jit, static_argnums=(6))
def THB_jacobian(
    params, ctrl_pts, fn_coeffs_indexed, repeat_ind, Jm, knotvectors, degrees
):
    jacobian = jacfwd(THB_evaluate, argnums=0)(
        params, ctrl_pts, fn_coeffs_indexed, repeat_ind, Jm, knotvectors, degrees
    )
    indices = jnp.arange(params.shape[0])
    output = jacobian[indices, :, indices, :]
    return output


def pre_process_data(
    params, fn_coeffs, ctrl_pts, ac_cells_supp, fn_shapes, knotvectors, degrees
):
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

    fn_coeffs_flat = jnp.vstack(fn_coeffs_flat)

    CP_dim = ctrl_pts[0].shape[-1]
    ctrl_pts_flat = [
        jnp.array(ctrl_pts[lev]).reshape(-1, CP_dim).astype(jnp.float32)
        for lev in range(max_lev + 1)
    ]
    ctrl_pts_stack = jnp.vstack(ctrl_pts_flat)

    Jm = jnp.array(
        [
            n_fns[fn_lev] + np.ravel_multi_index(fnIdx, fn_shapes[fn_lev])
            for cell_supp in ac_cells_supp
            for fn_lev, fnIdx in cell_supp
        ]
    )

    # def extract_slices(i, span):
    #     starts = span
    #     sizes = jnp.array(degrees) + 1
    #     return lax.dynamic_slice(fn_coeffs_per_param[i], starts, sizes)

    # vectorized_extraction = vmap(extract_slices, in_axes=(0, 0))

    # fn_coeffs_indexed = vectorized_extraction(
    #     jnp.arange(fn_coeffs_per_param.shape[0]), spans[Jm]
    # )
    fn_coeffs_indexed = THB_coeff_indexing(
        params, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees
    )
    return fn_coeffs_indexed, ctrl_pts_stack, Jm


def THB_coeff_indexing(
    params, ac_cells_supp, fn_coeffs, fn_shapes, knotvectors, degrees
):
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)

    spans = np.array(
        [
            np.array(
                find_span_array_jax(
                    params[:, dim], knotvectors[max_lev][dim], degrees[dim]
                )
            )
            for dim in range(len(degrees))
        ]
    ).T

    def basis_fn_worker(param_idx, param):
        # max_lev = max(knotvectors.keys())
        # ndim = len(degrees)
        cell_supp = ac_cells_supp[param_idx]
        max_lev_cellIdx = spans[param_idx]
        # max_lev_cellIdx = [
        #     findSpan(
        #         fn_shapes[max_lev][dim],
        #         degrees[dim],
        #         param[dim],
        #         knotvectors[max_lev][dim],
        #     )
        #     for dim in range(ndim)
        # ]

        slice_tuple = tuple(
            slice(max_lev_cellIdx[dim] - degrees[dim], max_lev_cellIdx[dim] + 1)
            for dim in range(ndim)
        )

        all_fn_coeffs = [
            fn_coeffs[fn_level][fnIdx][slice_tuple] for fn_level, fnIdx in cell_supp
        ]

        return np.stack(all_fn_coeffs)

    COEFF = [basis_fn_worker(i, g) for i, g in tqdm(enumerate(params))]

    return np.vstack(COEFF)
