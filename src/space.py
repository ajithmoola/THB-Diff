import numpy as np
from numpy import unravel_index as ind2sub
from numpy import ravel_multi_index as sub2ind
from numba import njit
from scipy.special import comb
from itertools import product
from functools import reduce
import operator
from funcs import assemble_Tmatrix, findSpan

class Space:

    def __init__(self, Hierarchy):
        self.Hierarchy = Hierarchy
        self.num_levels = max(self.Hierarchy.keys())
        self.dim = len(self.Hierarchy.H[0].bsplines)
        self.initialize_datastructures()
        self.compute_coefficients()

    def initialize_datastructures(self):
        self.knotvectors = {level:(bs.knotvector for bs in TP.bsplines) for level, TP in self.Hierarchy.H.items()}
        self.degrees = (bs.degree for bs in self.Hierarchy.H[0].bsplines)
        self.sh_knots = {level:(len(np.unique(self.knotvectors[level][i])) for i in range(self.dim)) for level in range(self.num_levels)}
        self.sh_cells = {level:(len(np.unique(self.knotvectors[level][i]))-1 for i in range(self.dim)) for level in range(self.num_levels)}
        self.sh_fns = {level:(self.sh_knots[i]+self.degrees[i] for i in range(self.dim)) for level in range(self.num_levels)}

        self.cells = {level:np.zeros(np.array(self.sh_knots[level])-1) for level in range(self.num_levels)}
        # self.fns = {level:np.zeros(self.sh_fns[level]) for level in range(self.num_levels)}
    
        self.cells[0] = np.ones_like(self.cells[0])
        # self.fns[0] = np.ones_like(self.fns[0])

    def compute_coefficients(self):

        self.Coeff = {}

        for lev in range(self.num_levels-1):
            for dim in range(self.dim):
                knotvector = self.knotvectors[lev][dim]
                refined_knotvector = self.knotvectors[lev+1][dim]
                self.Coeff[lev][dim] = assemble_Tmatrix(knotvector, refined_knotvector, knotvector.size, refined_knotvector.size, self.degrees[dim])
    
    def build_hierarchy_from_domain_sequence(self):
        # H0 = {beta in B0: supp(beta) not empty}
        H = {level:np.zeros(self.sh_fns) for level in range(self.num_levels)}
        H[0] = np.ones_like(H[0])

        # Recursively updating the basis functions to active or passive
        for lev in range(0, self.num_levels-1):
            
            num_fns = H[lev].size
            sh_fns = self.sh_fns[lev]

            # Iterating over all the basis functions in previous level to compute H_a
            deactivated_fns = []
            for idx in range(num_fns):
                # TODO: Should be able to run on GPU
                subIdx = ind2sub(idx, sh_fns)
                supp_cells = self.get_support_cell_indices(idx, lev)

                # Checking the status of support cells
                cell_bool = [True if self.cells[cell]==1 else False for cell in supp_cells]
                
                # deactivating functions whose support is not contained in O_l
                if np.all(np.array(cell_bool)==False):
                    H[lev][subIdx] = 0
                    deactivated_fns.append(subIdx)
            
            # Using simplified hierarchy definition computing Hb
            children = [self.get_children_fns(de_fn) for de_fn in deactivated_fns]
            H_b = set([item for sublist in children for item in sublist])

            for idx in H_b:
                H[lev+1][idx] = 1
        
        self.fns = H

    def compute_tensor_product(self, eval_pts_shape):
        
        self.pts = np.hstack(tuple(map(lambda x: x.reshape(-1, 1), np.meshgrid([np.linspace(1e-5, 1, eval_pts_shape[dim], endpoint=False) for dim in range(self.dim)]))))
        ac_cells_supp = {level:{} for level in range(self.num_levels)}

        for pt in self.pts:
            ac_cell, ac_lev = findActiveSpan()
            nonzero_fns = findSuppFns()

            if ac_cell not in ac_cells_supp[ac_lev].keys:
                ac_cells_supp[ac_lev][ac_cell] = self.compute_cell_active_support(ac_cell, ac_lev)

            ac_supp = ac_cells_supp[ac_lev][ac_cell]

            

    
    def compute_cell_active_support(self, cellSubIdx, level):

        active_supp_fns = {}

        for lev in range(level, 0, -1):

            fns = []
            for dim in range(self.dim):
                fns.append(sorted(set(cellSubIdx[dim]+i for i in range(self.degrees[dim]+1))))
            
            supp_fns = np.array(np.meshgrid(*fns, indexing='ij')).T.reshape(-1, len(self.dim))
            supp_fns = [tuple(row) for row in supp_fns]

            ac_fns = []
            for fn in supp_fns:
                if self.fns[lev][fn]==1:
                    ac_fns.append(fn)
            
            active_supp_fns[lev] = ac_fns

            # computing the parent cell index
            cellSubIdx = tuple(np.array(cellSubIdx)//2)

        return active_supp_fns
    
    def get_parent_cell(self, cellSubIdx):
        return tuple(np.array(cellSubIdx)//2)

    def s1_refine_cell(self, cellIndices):
        
        for cell, lev in cellIndices.items():

            multi_ind = ind2sub(cell, self.sh_cells[lev])
            self.cells[lev][multi_ind] = 0
            refined_indices = []

            # computing children indices
            for offset in product(range(2), repeat=len(multi_ind)):
                refined_indices.append(tuple(2*index + delta for index, delta in zip(multi_ind, offset)))

            refined_cells_shape = self.sh_cells[lev+1]
            # marking children as active
            for idx in refined_indices:
                self.cells[lev][sub2ind(idx, refined_cells_shape)] = 1

    def get_children_fns(self, fnIdx, level):

        subIdx = ind2sub(fnIdx)
        children = []

        for dim in range(self.dim):
            Coeff = self.Coeff[level][dim]
            children.append(Coeff[subIdx[dim]])

        grids = np.meshgrid(*children)
        combinations = np.stack(grids, axis=-1).reshape(-1, dim)

        return [tuple[row] for row in combinations]

    def get_support_cell_indices(self, fnIdx, level):
        multi_ind = ind2sub(fnIdx, self.sh_fns[level])
        return support_cells_multi(self.knotvectors[level], self.degrees, multi_ind)


@njit
def support_cells_multi(knot_vectors, degrees, indices):
    # TODO: write in C++ using pybind11
    all_support_cells = []

    for dim, (knot_vector, degree, i) in enumerate(zip(knot_vectors, degrees, indices)):
        start = max(i - degree, 0)
        end = min(i + 1, len(knot_vector) - degree - 1)

        cell_indices = set()
        for j in range(start, end):
            cell_indices.add(j)
        
        all_support_cells.append(sorted(cell_indices))

    support_cells_md = np.array(np.meshgrid(*all_support_cells, indexing='ij')).T.reshape(-1, len(degrees))

    return [tuple(cell) for cell in support_cells_md]

def findActiveSpan():
    pass

def findSuppFns():
    pass