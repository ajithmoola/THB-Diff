import numpy as np
from itertools import product
from copy import deepcopy
from typing import Dict, Tuple, List


from THB.core import get_children_fns, support_cells_multi
from THB.bspline_funcs import (
    refine_knotvector,
    compute_tensor_product,
    assemble_Tmatrix,
)


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


class HierarchicalSplineSpace:

    def __init__(self, tensor_product, num_levels):
        self.H = {0: tensor_product}
        self.degrees = tuple(bs.degree for bs in tensor_product.bsplines)
        for lev in range(1, num_levels):
            self.H[lev] = self.H[lev - 1].refine_tensorproduct()
        self.num_levels = num_levels
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

        self.cells = {
            level: np.zeros(self.sh_cells[level]) for level in range(self.num_levels)
        }

        self.cells[0] = np.ones_like(self.cells[0])

        self.Coeff = {}
        for lev in range(self.num_levels - 1):
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

    def build_hierarchy_from_domain_sequence(self):
        H = {level: np.zeros(self.sh_fns[level]) for level in range(self.num_levels)}
        H[0] = np.ones_like(H[0])

        for lev in range(0, self.num_levels - 1):
            deactivated_fns = []
            H_a = list(zip(*np.nonzero(H[lev])))
            for fn in H_a:
                supp_cells = self._support_cells_multi(lev, fn)
                cell_bool = [
                    True if self.cells[lev][cell] == 1.0 else False
                    for cell in supp_cells
                ]
                if np.all(np.array(cell_bool) == False):
                    H[lev][fn] = 0
                    deactivated_fns.append(fn)
            if not deactivated_fns:
                break
            children = [self._get_children_fns(de_fn, lev) for de_fn in deactivated_fns]
            H_b = set([item for sublist in children for item in sublist])
            for idx in H_b:
                H[lev + 1][idx] = 1

        self.fns = H

        self.fn_states.append(H)

    def compute_active_cells_active_supp(
        self,
    ) -> Dict[int, Dict[Tuple[int], List[Tuple[int]]]]:
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
            lev: compute_coeff_tensor_product(
                [self.Coeff[lev][dim] for dim in range(self.ndim)]
            )
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
                [coeffs_tp[lev + 1], projection_coeff[lev + 1]], ndim=self.ndim
            )

        # Truncation of basis functions
        for lev in range(max_lev - 1, -1, -1):
            curr_coeff_tp = deepcopy(coeffs_tp[lev])
            trunc_indices = np.where(self.fns[lev + 1] == 1)

            for fn in np.ndindex(self.fns[lev].shape):
                curr_coeff_tp[fn][trunc_indices] = 0

            if lev < (max_lev - 1):
                curr_coeff_tp = compute_projection(
                    [curr_coeff_tp, projection_coeff[lev]], ndim=self.ndim
                )

            fn_coeffs[lev] = np.array(curr_coeff_tp).astype(np.float16)

        fn_coeffs[max_lev] = np.ones(
            (*self.fns[max_lev].shape, *self.fns[max_lev].shape)
        )
        self.fn_coeffs = fn_coeffs

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
        """
        Get the children basis functions for a given basis function.

        Args:
            fnIdx (tuple): Index of the basis function.
            level (int): Level of the basis function.

        Returns:
            list: List of children basis functions.
        """
        children = []

        for dim in range(self.ndim):
            curr_coeff = self.Coeff[level][dim]
            children.append(np.nonzero(curr_coeff[fnIdx[dim]])[0])

        grids = np.meshgrid(*children)
        combinations = np.stack(grids, axis=-1).reshape(-1, self.ndim)

        return [tuple(row) for row in combinations]

    def _refine_basis_fn(self, fnIdx, level):
        assert level < self.num_levels - 1
        supp_cells = support_cells_multi(level, fnIdx)
        for cell in supp_cells:
            self._refine_cell(cell, level)

    def _refine_cell(self, cellIdx, level):
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
        self.cells[level - 1][cellIdx] = 1
        children_cells = []
        for offset in product(range(2), repeat=len(ancestor_cell)):
            children_cells.append(
                tuple(2 * index + delta for index, delta in zip(ancestor_cell, offset))
            )
        for child in children_cells:
            self.cells[level][child] = 0

    def _support_cells_multi(self, lev: int, fn: Tuple[int]) -> List[Tuple[int]]:
        """
        Compute the support cells for a given function.

        Args:
            knot_vectors (Dict): Dictionary of knot vectors for each dimension.
            fn (tuple): Tuple of indices for each dimension.

        Returns:
            list: List of support cells for the given function.
        """
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
        ).T.reshape(-1, len(self.ndim))

        return [tuple(cell) for cell in support_cells_md]


class ControlPoints:
    # Integrated into the THB_layer in eval.py
    def __init__(self, H_space):
        self.h_space = H_space
        self.max_lev = H_space.num_levels - 1
        self.ndim = H_space.ndim
        self.CP_status = {
            lev: np.zeros_like(H_space.fns[lev]) for lev in range(self.max_lev + 1)
        }
        self.CP_status[0] = np.ones_like(self.CP_status[0])

    def update_CP(self, CP_arr, fns, Coeffs):
        CP_arr = np.array(CP_arr)
        nCP = [0] + [
            np.prod(self.h_space.sh_fns[lev]) for lev in range(self.max_lev + 1)
        ]
        ctrl_pts = {
            lev: CP_arr[nCP[lev] : nCP[lev + 1]].reshape(
                *self.h_space.sh_fns[lev], self.ndim
            )
            for lev in range(self.max_lev + 1)
        }

        for lev in range(1, self.max_lev + 1):
            curr_coeff = Coeffs[lev - 1]
            for CP in np.ndindex(fns[lev].shape):
                if self.CP_status[lev][CP] == 0 and fns[lev][CP] == 1:
                    cp_coeff = [curr_coeff[dim].T[CP[dim]] for dim in range(self.ndim)]
                    tp = compute_tensor_product(cp_coeff)
                    ctrl_pts[lev][CP] = np.sum(
                        tp[..., np.newaxis] * ctrl_pts[lev - 1],
                        axis=tuple(range(len(tp.shape))),
                    )
        self.CP_status = deepcopy(fns)
        return ctrl_pts
