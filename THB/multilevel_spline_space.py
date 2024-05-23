import numpy as np
from itertools import product
from copy import deepcopy

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


class Space:

    def __init__(self, tensor_product, num_levels):
        self.H = {0: tensor_product}
        self.degrees = tuple(bs.degree for bs in tensor_product.bsplines)
        for lev in range(1, num_levels):
            self.H[lev] = self.H[lev - 1].refine_tensorproduct()
        self.num_levels = num_levels
        self.ndim = len(tensor_product.bsplines)
        self.fn_states = []
        self.initialize_datastructures()
        self.compute_coefficients()

    def initialize_datastructures(self):
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

    def compute_coefficients(self):
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
                supp_cells = support_cells_multi(
                    self.knotvectors[lev], self.degrees, fn
                )
                cell_bool = [
                    True if self.cells[lev][cell] == 1.0 else False
                    for cell in supp_cells
                ]
                if np.all(np.array(cell_bool) == False):
                    H[lev][fn] = 0
                    deactivated_fns.append(fn)
            if not deactivated_fns:
                break
            children = [
                get_children_fns(de_fn, self.Coeff, lev, self.ndim)
                for de_fn in deactivated_fns
            ]
            H_b = set([item for sublist in children for item in sublist])
            for idx in H_b:
                H[lev + 1][idx] = 1

        self.fns = H

        self.fn_states.append(H)

    def _refine_basis_fn(self, fnIdx, level):
        assert level < self.num_levels - 1
        supp_cells = support_cells_multi(self.knotvectors[level], self.degrees, fnIdx)
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
