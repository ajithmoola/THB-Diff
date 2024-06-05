import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from time import time
from tqdm import tqdm
from typing import Dict, Tuple, List


# from OCC.Core.Geom import Geom_BSplineSurface
# from OCC.Core.TColgp import TColgp_Array2OfPnt
# from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
# from OCC.Core.gp import gp_Pnt
# from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
# from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

from THB.bspline_funcs import *


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def CP_arr_to_dict(CP_arr, sh_fns, num_levels):
    CP_arr = np.array(CP_arr)
    nCP = np.array([0] + [np.prod(sh_fns[lev]) for lev in range(num_levels)]).cumsum()
    ctrl_pts = {
        lev: CP_arr[nCP[lev] : nCP[lev + 1]].reshape(*sh_fns[lev], CP_arr.shape[1])
        for lev in range(num_levels)
    }
    return ctrl_pts


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


# def BSplineSurf_to_STEP(CP, knotvectors, degrees, fname):
#     """Exports maximum level b-spline to a step file

#     Args:
#         CP (ndarray): control points
#         knotvectors (list): knotvectors in a tuple
#         degrees (tuple): degree of b-splines in the tensor product
#         fname (str): file name
#     """
#     degree_u = degrees[0]
#     degree_v = degrees[1]

#     knots_u = np.unique(knotvectors[0])
#     knots_v = np.unique(knotvectors[1])

#     multiplicities_u = np.ones_like(knots_u)
#     multiplicities_v = np.ones_like(knots_v)
#     multiplicities_u[0] = degree_u + 1
#     multiplicities_u[-1] = degree_u + 1
#     multiplicities_v[0] = degree_u + 1
#     multiplicities_v[-1] = degree_v + 1

#     knots_u_occ = TColStd_Array1OfReal(1, len(knots_u))
#     knots_v_occ = TColStd_Array1OfReal(1, len(knots_v))

#     multiplicities_u_occ = TColStd_Array1OfInteger(1, len(multiplicities_u))
#     multiplicities_v_occ = TColStd_Array1OfInteger(1, len(multiplicities_v))

#     for i, val in enumerate(knots_u, start=1):
#         knots_u_occ.SetValue(i, val)
#     for i, val in enumerate(knots_v, start=1):
#         knots_v_occ.SetValue(i, val)
#     for i, val in enumerate(multiplicities_u):
#         multiplicities_u_occ.SetValue(i + 1, int(val))
#     for i, val in enumerate(multiplicities_v):
#         multiplicities_v_occ.SetValue(i + 1, int(val))

#     control_points_occ = TColgp_Array2OfPnt(1, CP.shape[0], 1, CP.shape[1])
#     for i in range(CP.shape[0]):
#         for j in range(CP.shape[1]):
#             x, y, z = map(float, CP[i, j])
#             control_points_occ.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))

#     bspline_surface = Geom_BSplineSurface(
#         control_points_occ,
#         knots_u_occ,
#         knots_v_occ,
#         multiplicities_u_occ,
#         multiplicities_v_occ,
#         degree_u,
#         degree_v,
#         False,
#         False,
#     )

#     face = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6).Face()

#     writer = STEPControl_Writer()
#     writer.Transfer(face, STEPControl_AsIs)

#     status = writer.Write(fname + ".step")

#     if status:
#         print("Successfully exported B-spline surface to STEP file.")
#     else:
#         print("Failed to export B-spline surface to STEP file.")
