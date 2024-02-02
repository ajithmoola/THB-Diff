import numpy as np
from funcs import *
from copy import deepcopy

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

class ControlPoints:
    def __init__(self, H_space):
        self.H = H_space.H
        self.num_levels = H_space.num_levels
        self.ndim = H_space.ndim
        self.CP_status = {lev: np.zeros_like(H_space.fns[lev]) for lev in range(H_space.num_levels)}
        self.CP_status[0] = np.ones_like(self.CP_status[0])
    
    def update_CP(self, ctrl_pts, fns, Coeffs):
        for lev in range(1, self.num_levels):
            curr_coeff = Coeffs[lev-1]
            for CP in np.ndindex(fns[lev].shape):
                if self.CP_status[lev][CP]==0 and fns[lev][CP]==1:
                    cp_coeff = [curr_coeff[dim].T[CP[dim]] for dim in range(self.ndim)]
                    tp = compute_tensor_product(cp_coeff)
                    ctrl_pts[lev][CP] = np.sum(tp[..., np.newaxis]*ctrl_pts[lev-1], axis=tuple(range(len(tp.shape))))
        self.CP_status = deepcopy(fns)
        return ctrl_pts

class Octree:

    def __init__(self, bounds, dim):
        # self.index = index
        self.bounds = bounds
        self.children = []
    
    def refine(self):
        x_min, y_min, z_min, x_max, y_max, z_max = self.bounds

        mid_x = (x_min + x_max) / 2
        mid_y = (y_min + y_max) / 2
        mid_z = (z_min + z_max) / 2

        self.children = [
            Octree(())
        ]