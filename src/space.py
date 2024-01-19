import numpy as np
from numpy import unravel_index as ind2sub
from numpy import ravel_multi_index as sub2ind
from numba import njit
from scipy.special import comb
from itertools import product
from functools import reduce
import operator
from funcs import assemble_Tmatrix, findSpan
from THB_utils import get_children_fns

class Space:

    def __init__(self, Hierarchy):
        self.Hierarchy = Hierarchy
        self.num_levels = len(self.Hierarchy.H.keys())
        self.dim = len(self.Hierarchy.H[0].bsplines)
        self.initialize_datastructures()
        self.compute_coefficients()

    def initialize_datastructures(self):
        self.knotvectors = {level:tuple(bs.knotvector for bs in TP.bsplines) for level, TP in self.Hierarchy.H.items()}
        self.degrees = tuple(bs.degree for bs in self.Hierarchy.H[0].bsplines)
        self.sh_knots = {level:tuple(len(np.unique(self.knotvectors[level][i])) for i in range(self.dim)) for level in range(self.num_levels)}
        self.sh_cells = {level:tuple(len(np.unique(self.knotvectors[level][i]))-1 for i in range(self.dim)) for level in range(self.num_levels)}
        self.sh_fns = {level:tuple(self.sh_cells[level][i]+self.degrees[i] for i in range(self.dim)) for level in range(self.num_levels)}

        self.cells = {level:np.zeros(self.sh_cells[level]) for level in range(self.num_levels)}
        # self.fns = {level:np.zeros(self.sh_fns[level]) for level in range(self.num_levels)}
    
        self.cells[0] = np.ones_like(self.cells[0])
        # self.fns[0] = np.ones_like(self.fns[0])

    def compute_coefficients(self):
        self.Coeff = {}
        for lev in range(self.num_levels-1):
            curr_coeff = {}
            for dim in range(self.dim):
                knotvector = self.knotvectors[lev][dim]
                refined_knotvector = self.knotvectors[lev+1][dim]
                curr_coeff[dim] = assemble_Tmatrix(knotvector, refined_knotvector, knotvector.size, refined_knotvector.size, self.degrees[dim]).T
            self.Coeff[lev] = curr_coeff
    
    def build_hierarchy_from_domain_sequence(self):
        # tested: working! (not sure 100%)
        # H0 = {beta in B0: supp(beta) not empty}
        H = {level:np.zeros(self.sh_fns[level]) for level in range(self.num_levels)}
        H[0] = np.ones_like(H[0])

        # Recursively updating the basis functions to active or passive
        for lev in range(0, self.num_levels-1):
            # Iterating over all the basis functions in previous level to compute H_a
            deactivated_fns = []
            H_a = list(zip(*np.nonzero(H[lev])))
            for fn in H_a:
                supp_cells = self.get_support_cell_indices(fn, lev)
                # Checking the status of support cells
                cell_bool = [True if self.cells[lev][cell]==1.0 else False for cell in supp_cells]
                # deactivating functions whose support is not contained in O_l
                if np.all(np.array(cell_bool)==False):
                    H[lev][fn] = 0
                    deactivated_fns.append(fn)
            # If none of the functions are deactivated then break the loop
            if not deactivated_fns:
                break
            # TODO: any use for deactivated functions information?
            # Using simplified hierarchy definition computing Hb
            children = [get_children_fns(de_fn, self.Coeff, lev, self.dim) for de_fn in deactivated_fns]
            # print(children)
            H_b = set([item for sublist in children for item in sublist])
            for idx in H_b:
                H[lev+1][idx] = 1
        
        self.fns = H
    
    def refine_cell(self, cellIdx, level):
        self.cells[level][cellIdx] = 0
        children_cells = []
        for offset in product(range(2), repeat=len(cellIdx)):
            children_cells.append(tuple(2*index + delta for index, delta in zip(cellIdx, offset)))
        for child in children_cells:
            self.cells[level+1][child] = 1

    def get_support_cell_indices(self, fnIdx, level):
        return support_cells_multi(self.knotvectors[level], self.degrees, fnIdx)

def support_cells_multi(knot_vectors, degrees, fn):
    # TODO: write in C++ using pybind11
    # tested: working! (but not 100% sure)
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