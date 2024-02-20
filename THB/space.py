import numpy as np
from itertools import product
from THB.funcs import assemble_Tmatrix
from THB.core import *


class Space:

    def __init__(self, tensor_product, num_levels):
        self.H = {0: tensor_product}
        self.degrees = tuple(bs.degree for bs in tensor_product.bsplines)
        for lev in range(1, num_levels):
            self.H[lev] = self.H[lev - 1].refine_tensorproduct()
        self.num_levels = num_levels
        self.ndim = len(tensor_product.bsplines)
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
                supp_cells = self.get_support_cell_indices(fn, lev)
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

    def compute_coefficients(self):
        self.ac_cells = compute_active_cells_active_supp(
            self.cells, self.fns, self.degrees
        )
        self.fn_coeffs = compute_refinement_operators(
            self.fns, self.Coeff, self.degrees
        )

    def set_parameters(self, parameters):
        self.parameters = parameters

    def compute_tensor_product(self):
        self.ac_spans = compute_active_span(
            self.parameters, self.knotvectors, self.cells, self.degrees, self.sh_fns
        )
        self.PHI, self.num_supp = compute_basis_fns_tp_parallel(
            self.parameters,
            self.ac_spans,
            self.ac_cells,
            self.fn_coeffs,
            self.sh_fns,
            self.knotvectors,
            self.degrees,
        )

    def refine_basis_fn(self, fnIdx, level):
        assert level < self.num_levels - 1
        supp_cells = self.get_support_cell_indices(fnIdx, level)
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

    def get_support_cell_indices(self, fnIdx, level):
        return support_cells_multi(self.knotvectors[level], self.degrees, fnIdx)
