from itertools import product
from copy import deepcopy
from typing import Dict, Tuple, List
from functools import partial

import pyvista as pv
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from THBDiff.core_funcs import (
    refine_knotvector,
    assemble_Tmatrix,
    grevilleAbscissae,
    find_span_array,
)
from THBDiff.bspline_2D import compute_tensor_product_full, compute_basis_at_point_full


class BSpline:

    def __init__(self, knotvector, degree):
        self.knotvector = knotvector
        self.degree = degree

    def refine_bspline(self):
        new_knotvector = refine_knotvector(self.knotvector, self.degree)
        self.next_level = BSpline(new_knotvector, self.degree)
        return self.next_level


class TensorProduct:
    def __init__(self, bsplines):
        self.bsplines = bsplines

    def refine_tensorproduct(self):
        self.next_level = TensorProduct([bs.refine_bspline() for bs in self.bsplines])
        return self.next_level


@partial(jit, static_argnums=(5))
def THB_basis_fns(
    params: jnp.ndarray,
    p_coeff: jnp.ndarray,
    repeat_ind: List[int],
    Jm,
    max_knotvectors: Tuple[np.ndarray],
    degrees: Tuple[int],
) -> jnp.ndarray:
    basis_fns_tp = compute_tensor_product_full(params, max_knotvectors, degrees)
    PHI = jnp.einsum("i... -> i", basis_fns_tp[repeat_ind] * p_coeff[Jm])[:, None]
    return PHI


def Evaluate_JAX(ctrl_pts, Jm, PHI, segment_ids, num_pts, cp_dim):
    prod = PHI * ctrl_pts[Jm]
    output = jnp.zeros((num_pts, cp_dim)).at[segment_ids].add(prod)
    return output


def evaluate_single_param(
    param_single,
    ctrl_pts_indexed,
    p_coeff_indexed,
    maxknotvectors,
    degrees,
    num_supp_single,
    num_supp_ll,
    num_supp_ul,
    total_repeat_length,
):
    basis_fns_tp_single = compute_basis_at_point_full(
        param_single, maxknotvectors, degrees
    )[None, :, :]
    b_fns_tp_repeated = jnp.repeat(
        basis_fns_tp_single,
        num_supp_single,
        axis=0,
        total_repeat_length=total_repeat_length,
    )
    p_coeff_indexed_nonzero = p_coeff_indexed[num_supp_ll:num_supp_ul]
    ctrl_pts_indexed_nonzero = ctrl_pts_indexed[num_supp_ll:num_supp_ul]
    point = (
        ((b_fns_tp_repeated * p_coeff_indexed_nonzero).sum(axis=(1, 2))[:, None])
        * ctrl_pts_indexed_nonzero
    ).sum(axis=(0))
    return point


def evaluate_vmapped(
    params, ctrl_pts, p_coeff, Jm, num_supp, num_supp_limits, max_kv, degrees
):
    ctrl_pts_indexed = ctrl_pts[Jm]
    p_coeff_indexed = p_coeff[Jm]
    total_repeat_length = num_supp.sum()
    num_supp_ll_all = num_supp_limits[:, 0]
    num_supp_ul_all = num_supp_limits[:, 1]
    pts = vmap(
        evaluate_single_param, in_axes=(0, None, None, None, None, 0, 0, 0, None)
    )(
        params,
        ctrl_pts_indexed,
        p_coeff_indexed,
        max_kv,
        degrees,
        num_supp,
        num_supp_ll_all,
        num_supp_ul_all,
        total_repeat_length,
    )
    return pts


def evaluate(
    params,
    ctrl_pts,
    p_coeff,
    maxknotvectors,
    degrees,
    Jm,
    segment_ids,
):
    # Num of fns (N) = Num of params * number of non-zero support funs per param
    # basis_fns = (n, 22, 22), basis_fns[segment_ids] = (N, 22, 22)
    # fn_coeffs = (677, 22, 22), fn_coeffs[Jm] = (N, 22, 22)
    # ctrl_pts = (677, 3, 3), ctrl_pts[Jm] = (N, 3, 3)
    num_pts = params.shape[0]
    cp_dim = ctrl_pts.shape[-1]
    basis_fns = compute_tensor_product_full(params, maxknotvectors, degrees)
    PHI = jnp.einsum("i... -> i", basis_fns[segment_ids] * p_coeff[Jm])[:, None]
    prod = PHI * ctrl_pts[Jm]
    output = jnp.zeros((num_pts, cp_dim)).at[segment_ids].add(prod)
    return output


def compute_thb_jacobian(
    params, ctrl_pts, p_coeff, maxknotvectors, degrees, Jm, repeat_ind
):
    thb_jac_func = jacfwd(evaluate, argnums=(0,))
    thb_jac = thb_jac_func(
        params, ctrl_pts, p_coeff, maxknotvectors, degrees, Jm, repeat_ind
    )
    indices = jnp.arange(params.shape[0])
    jac_indexed = thb_jac[indices, :, indices, :]
    return jac_indexed


class HierarchicalSplineSpace:

    def __init__(self, tensor_product, num_levels):
        self.H = {0: tensor_product}
        self.degrees = tuple(bs.degree for bs in tensor_product.bsplines)
        for lev in range(1, num_levels):
            self.H[lev] = self.H[lev - 1].refine_tensorproduct()
        self.num_levels = num_levels
        self.max_lev = self.num_levels - 1
        self.ndim = len(tensor_product.bsplines)
        self.fn_states = []

        self.knotvectors = {
            level: tuple(bs.knotvector for bs in TP.bsplines)
            for level, TP in self.H.items()
        }

        self.sh_knots = {
            level: tuple(
                len(np.unique(self.knotvectors[level][i])) for i in range(self.ndim)
            )
            for level in range(self.num_levels)
        }
        self.sh_cells = {
            level: tuple(
                len(np.unique(self.knotvectors[level][i])) - 1 for i in range(self.ndim)
            )
            for level in range(self.num_levels)
        }
        self.sh_fns = {
            level: tuple(
                self.sh_cells[level][i] + self.degrees[i] for i in range(self.ndim)
            )
            for level in range(self.num_levels)
        }

        self.n_fns = sum(np.prod(self.sh_fns[lev]) for lev in range(self.num_levels))

        self.cells = {
            level: np.zeros(self.sh_cells[level]) for level in range(self.num_levels)
        }

        self.cells[0] = np.ones_like(self.cells[0])

        self.Coeff = {}
        for lev in range(self.max_lev):
            curr_coeff = {}
            for dim in range(self.ndim):
                knotvector = self.knotvectors[lev][dim]
                refined_knotvector = self.knotvectors[lev + 1][dim]
                curr_coeff[dim] = assemble_Tmatrix(
                    knotvector,
                    refined_knotvector,
                    knotvector.size,
                    refined_knotvector.size,
                    self.degrees[dim],
                ).T
            self.Coeff[lev] = curr_coeff
        self.refinement_history = []

        self.GA = {}
        for lev in range(num_levels):
            self.GA[lev] = grevilleAbscissae(
                self.sh_fns[lev], self.degrees, self.knotvectors[lev]
            )

    def get_zero_ctrl_pts(self):
        return {
            lev: jnp.ones((*(self.sh_fns[lev]), self.ndim))
            for lev in range(self.num_levels)
        }

    def build_hierarchy_from_domain_sequence(self):
        refined = {self.max_lev: np.zeros(self.sh_cells[self.max_lev], dtype=bool)}
        omega = {self.max_lev: self.cells[self.max_lev] == 1}
        for lev in range(self.max_lev - 1, -1, -1):
            child_shape = [s for n in self.sh_cells[lev] for s in (n, 2)]
            refined[lev] = omega[lev + 1].reshape(child_shape).any(
                axis=tuple(range(1, 2 * self.ndim, 2))
            )
            omega[lev] = (self.cells[lev] == 1) | refined[lev]

        H = {level: np.zeros(self.sh_fns[level]) for level in range(self.num_levels)}
        for lev in range(self.num_levels):
            for fn in np.ndindex(self.sh_fns[lev]):
                supp_cells = self._support_cells_multi(lev, fn)
                in_omega = all(omega[lev][cell] for cell in supp_cells)
                fully_refined = all(refined[lev][cell] for cell in supp_cells)
                if in_omega and not fully_refined:
                    H[lev][fn] = 1

        self.fns = H

        self.refinement_history.append(deepcopy(self.fns))
        if len(self.refinement_history) > 1:
            self.GA = self.update_CP(self.GA)
        self.compute_active_cells_active_supp()
        self.compute_refinement_operators()

    def compute_active_cells_active_supp(
        self,
    ):
        """
        Compute the active cells and their active supports.
        """
        ac_cells = {}

        for lev in range(self.num_levels):
            curr_ac_cells = list(zip(*np.nonzero(self.cells[lev])))
            curr_lev_ac_cells_ac_supp = {}
            for cell in curr_ac_cells:
                curr_lev_ac_cells_ac_supp[cell] = self._compute_cell_active_supp(
                    cell, lev
                )
            ac_cells[lev] = curr_lev_ac_cells_ac_supp

        self.ac_cells = ac_cells

    def compute_refinement_operators(self):
        """
        Compute the projection matrices for the basis functions.
        """
        max_lev = self.num_levels - 1

        fn_projection_coeffs = {lev: {} for lev in range(max_lev + 1)}

        # Computes the tensor product of the projection coefficients from i-th level to (i+1)-th level
        # For 2D tensorproduct THB-splines the transformation can be implemented as an einsum
        # (ij, kl) -> ikjl
        # For 3D tensorproduct THB-splines
        # (ij, kl, mn) -> ikmjln

        def compute_coeff_tensor_product(args):
            return np.einsum("ij, kl -> ikjl", *args, optimize=True)

        coeffs_tp = {
            lev: compute_coeff_tensor_product(
                [self.Coeff[lev][dim] for dim in range(self.ndim)]
            )
            for lev in range(max_lev)
        }

        # Truncation of basis functions against every finer level
        for lev in range(max_lev - 1, -1, -1):
            curr_coeff_tp = deepcopy(coeffs_tp[lev])
            curr_coeff_tp[..., self.fns[lev + 1] == 1] = 0
            for fine_lev in range(lev + 1, max_lev):
                curr_coeff_tp = np.einsum(
                    "ijkl, klmn -> ijmn", *[curr_coeff_tp, coeffs_tp[fine_lev]]
                )
                curr_coeff_tp[..., self.fns[fine_lev + 1] == 1] = 0

            fn_projection_coeffs[lev] = np.array(curr_coeff_tp).astype(np.float32)

        max_lev_coeffs = [
            np.eye(self.sh_fns[max_lev][dim], dtype=np.float32)
            for dim in range(self.ndim)
        ]
        max_lev_coeffs_tp = compute_coeff_tensor_product(max_lev_coeffs)
        fn_projection_coeffs[max_lev] = max_lev_coeffs_tp
        self.fn_projection_coeffs = fn_projection_coeffs

    def pre_process_data(self, params):

        spans = {}
        active_cell_supp = []
        num_supp = []

        for lev in range(self.max_lev + 1):
            curr_spans = np.array(
                [
                    find_span_array(
                        params[:, dim], self.knotvectors[lev][dim], self.degrees[dim]
                    )
                    - self.degrees[dim]
                    for dim in range(self.ndim)
                ]
            ).T
            spans[lev] = curr_spans

        for i in tqdm(range(len(params))):
            for lev in range(self.max_lev, -1, -1):
                cellIdx = spans[lev][i]
                cellIdx = tuple(map(int, cellIdx))
                if self.cells[lev][cellIdx] == 1:
                    cell_supp = self.ac_cells[lev][cellIdx]
                    active_cell_supp.append(cell_supp)
                    num_supp.append(len(cell_supp))
                    break

        self.ac_cell_supp = active_cell_supp
        n_fns = np.zeros(self.max_lev + 2, dtype=np.int32)
        for lev in range(1, self.max_lev + 2):
            n_fns[lev] = n_fns[lev - 1] + np.prod(self.sh_fns[lev - 1])

        repeat_ind = jnp.repeat(jnp.arange(params.shape[0]), np.array(num_supp)).astype(
            jnp.int32
        )

        Jm = jnp.array(
            [
                n_fns[fn_lev] + np.ravel_multi_index(fnIdx, self.sh_fns[fn_lev])
                for cell_supp in active_cell_supp
                for fn_lev, fnIdx in cell_supp
            ]
        )

        return repeat_ind, Jm

    def _compute_cell_active_supp(
        self, cellIdx: Tuple[int], curr_level: int
    ) -> List[Tuple[int]]:
        """
        Compute the active supports for a given cell.

        Args:
            cellIdx (tuple): Index of the cell.
            curr_level (int): Current level.

        Returns:
            list: List of active supports for the given cell.
        """
        ac_supp = []
        for lev in range(curr_level, -1, -1):
            supp = self._get_supp_fns(cellIdx)
            for fn in supp:
                if self.fns[lev][fn] == 1:
                    ac_supp.append((lev, fn))
            cellIdx = tuple(np.array(cellIdx) // 2)
        return ac_supp

    def _get_supp_fns(self, cellIdx: Tuple[int]) -> List[Tuple[int]]:
        """
        Get the support basis functions for a given cell.

        Args:
            cellIdx (tuple): Index of the cell.
            degrees (tuple): Tuple of degrees for each dimension.

        Returns:
            list: List of support basis functions.
        """
        ranges = [range(idx, idx + p + 1) for idx, p in zip(cellIdx, self.degrees)]
        return [basisIdx for basisIdx in product(*ranges)]

    def _get_children_fns(
        self,
        fnIdx: Tuple[int],
        level: int,
    ) -> List[Tuple[int]]:
        children = []
        tolerance = 1e-6

        for dim in range(self.ndim):
            curr_coeff = self.Coeff[level][dim]
            children.append(np.nonzero(np.abs(curr_coeff[fnIdx[dim]]) > tolerance)[0])

        grids = np.meshgrid(*children)
        combinations = np.stack(grids, axis=-1).reshape(-1, self.ndim)

        return [tuple(row) for row in combinations]

    def _refine_basis_fn(self, fnIdx, level):
        assert level < self.num_levels - 1
        supp_cells = self._support_cells_multi(level, fnIdx)
        for cell in supp_cells:
            self._refine_cell(cell, level)

    def _refine_cell(self, cellIdx, level):
        if self.cells[level][cellIdx] != 1:
            return
        self.cells[level][cellIdx] = 0
        children_cells = []
        for offset in product(range(2), repeat=len(cellIdx)):
            children_cells.append(
                tuple(2 * index + delta for index, delta in zip(cellIdx, offset))
            )
        for child in children_cells:
            self.cells[level + 1][child] = 1

    def _collapse_cell(self, cellIdx, level):
        ancestor_cell = tuple(np.array(cellIdx) // 2)
        self.cells[level - 1][ancestor_cell] = 1
        children_cells = []
        for offset in product(range(2), repeat=len(ancestor_cell)):
            children_cells.append(
                tuple(2 * index + delta for index, delta in zip(ancestor_cell, offset))
            )
        for child in children_cells:
            self.cells[level][child] = 0

    def _support_cells_multi(self, lev: int, fn: Tuple[int]) -> List[Tuple[int]]:
        all_support_cells = []

        knot_vectors = self.knotvectors[lev]

        for dim, (knot_vector, degree, i) in enumerate(
            zip(knot_vectors, self.degrees, fn)
        ):
            start = max(i - degree, 0)
            end = min(i + 1, len(np.unique(knot_vector)) - 1)
            cell_indices = set()
            for j in range(start, end):
                cell_indices.add(j)

            all_support_cells.append(sorted(cell_indices))

        support_cells_md = np.array(
            np.meshgrid(*all_support_cells, indexing="ij")
        ).T.reshape(-1, self.ndim)

        return [tuple(cell) for cell in support_cells_md]

    def _get_control_grid_points_2D(self):
        knots = {
            lev: {
                dim: np.unique(self.knotvectors[lev][dim]) for dim in range(self.ndim)
            }
            for lev in range(self.num_levels)
        }
        cell_corners = []
        for lev in range(self.num_levels):
            for cell in self.ac_cells[lev].keys():
                x1 = knots[lev][0][cell[0]]
                y1 = knots[lev][1][cell[1]]
                x2 = knots[lev][0][cell[0] + 1]
                y2 = knots[lev][1][cell[1] + 1]
                if y2 == 1.0:
                    y2 -= 1e-7
                if x2 == 1.0:
                    x2 -= 1e-7

                if y1 == 1.0:
                    y1 -= 1e-7
                if x1 == 1.0:
                    x1 -= 1e-7

                # VTK hexahedron point ordering (bottom face counter-clockwise, top face counter-clockwise)
                # Bottom face: 0,1,2,3 (z=z1)
                # Top face: 4,5,6,7 (z=z2)
                p0 = [x1, y1]  # bottom-left-back
                p1 = [x2, y1]  # bottom-right-back
                p2 = [x2, y2]  # bottom-right-front
                p3 = [x1, y2]  # bottom-left-front
                cell_corners.append(np.array([p0, p1, p2, p3]).reshape(-1, 2))
        return np.vstack(cell_corners)

    def plot_adaptive_grid_2D(self, filepath):
        ctrl_pts = self.CP_dict_to_arr(self.GA)
        params = self._get_control_grid_points_2D()
        repeat_ind, Jm = self.pre_process_data(params)
        p_coeff = self.get_projection_coeffs_arr()
        PHI = THB_basis_fns(
            params,
            p_coeff,
            repeat_ind,
            Jm,
            self.knotvectors[self.max_lev],
            self.degrees,
        )
        pts = np.array(
            evaluate(
                params,
                ctrl_pts,
                p_coeff,
                self.knotvectors[self.max_lev],
                self.degrees,
                Jm,
                repeat_ind,
            )
        )
        fig, ax = plt.subplots()
        for i in range(0, len(pts), 4):
            cell_pts = pts[i : i + 4]

            rectangle = np.array(
                [cell_pts[0], cell_pts[1], cell_pts[2], cell_pts[3], cell_pts[0]]
            )
            ax.plot(rectangle[:, 0], rectangle[:, 1], color="black", linewidth=0.5)
        # ax.scatter(pts[:, 0], pts[:, 1])
        # ax.scatter(
        #     params[:, 0], params[:, 1], facecolors="none", marker="o", color="red"
        # )
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.show()

    def get_projection_coeffs_arr(self):
        max_lev_shape = self.sh_fns[self.max_lev]
        p_coeff_arr = np.vstack(
            [
                self.fn_projection_coeffs[lev].reshape(-1, *max_lev_shape)
                for lev in range(self.num_levels)
            ]
        )
        return p_coeff_arr

    def CP_arr_to_dict(self, CP_arr):
        CP_arr = np.array(CP_arr)
        CP_dim = CP_arr.shape[-1]
        nCP = np.array(
            [0] + [np.prod(self.sh_fns[lev]) for lev in range(self.num_levels)]
        ).cumsum()
        ctrl_pts = {
            lev: CP_arr[nCP[lev] : nCP[lev + 1]].reshape(*self.sh_fns[lev], CP_dim)
            for lev in range(self.num_levels)
        }
        return ctrl_pts

    @staticmethod
    def CP_dict_to_arr(CP_dict):
        num_levels = len(CP_dict)
        CP_arr = np.vstack(
            [
                CP_dict[lev].reshape(-1, CP_dict[lev].shape[-1])
                for lev in range(num_levels)
            ]
        )
        return CP_arr

    def update_CP(self, ctrl_pts):
        if not isinstance(ctrl_pts, dict):
            ctrl_pts = self.CP_arr_to_dict(np.array(ctrl_pts))
        else:
            ctrl_pts = self.CP_arr_to_dict(np.array(self.CP_dict_to_arr(ctrl_pts)))
        CP_status = self.refinement_history[-2]
        for lev in range(1, self.num_levels):
            curr_coeff = self.Coeff[lev - 1]
            for CP in np.ndindex(self.fns[lev].shape):
                if CP_status[lev][CP] == 0 and self.fns[lev][CP] == 1:
                    cp_coeff = [curr_coeff[dim].T[CP[dim]] for dim in range(self.ndim)]
                    tp = np.einsum("i, j -> ij", *cp_coeff)
                    ctrl_pts[lev][CP] = np.sum(
                        tp[..., np.newaxis] * ctrl_pts[lev - 1],
                        axis=tuple(range(len(tp.shape))),
                    )

        return self.CP_arr_to_dict(jnp.array(self.CP_dict_to_arr(ctrl_pts)))


def refine_thb(thb, error_values, ac_cell_supp, thresold_value, thresold_type):
    if thresold_type == "constant":
        high_error_points = error_values >= thresold_value
    elif thresold_type == "percentile":
        high_error_points = error_values >= jnp.percentile(error_values, thresold_value)

    high_error_indices = jnp.argwhere(high_error_points).squeeze().tolist()

    basis_fns_to_refine = set()
    print(high_error_indices)
    for index in high_error_indices:
        supp_fns = ac_cell_supp[index]
        basis_fns_to_refine.update(supp_fns)

    for thb_fn in basis_fns_to_refine:
        fn_level, fn_idx = thb_fn
        thb._refine_basis_fn(fn_idx, fn_level)

    thb.build_hierarchy_from_domain_sequence()

    return thb


if __name__ == "__main__":
    bs1 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)
    bs2 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)

    tp = TensorProduct([bs1, bs2])
    h_space = HierarchicalSplineSpace(tp, num_levels=3)
    h_space.build_hierarchy_from_domain_sequence()
    h_space._refine_basis_fn((5, 5), 0)
    h_space._refine_basis_fn((2, 2), 0)
    h_space.build_hierarchy_from_domain_sequence()
    h_space._refine_basis_fn((2, 2), 1)
    h_space.build_hierarchy_from_domain_sequence()
    h_space.plot_adaptive_grid_2D("thb_diff_lvl2.png")
