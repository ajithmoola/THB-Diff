import numpy as np
from funcs import refine_knotvector

class BSpline:

    def __init__(self, knotvector, degree):
        # tested: working!
        self.knotvector = knotvector
        self.degree = degree
    
    def refine_bspline(self):
        new_knotvector = refine_knotvector(self.knotvector, self.degree)
        self.next_level = BSpline(new_knotvector, self.degree)
        return self.next_level


class TensorProduct:
    # tested: working!
    def __init__(self, bsplines):
        self.bsplines = bsplines
    
    def refine_tensorproduct(self):
        self.next_level = TensorProduct([bs.refine_bspline() for bs in self.bsplines])
        return self.next_level

# class Hierarchy:
#     #tested: working!
#     def __init__(self, InitialSpace):
#         self.H = {0: InitialSpace}
#         self.degrees = (bs.degree for bs in InitialSpace.bsplines)
#         self.max_level = 0
    
#     def construct_multilevel_sequence(self, num_levels):
#         self.num_levels = num_levels
#         for lev in range(1, num_levels):
#             self.H[lev] = self.H[lev-1].refine_tensorproduct()

#     def add_level(self):
#         refined_space = self.H[self.max_level]
#         self.max_level += 1
#         self.num_levels += 1
#         self.H[self.max_level] = refined_space.refine_tensorproduct()

class ControlPoints:
    def __init__(self, H_space):
        self.H = H_space.H
        self.num_levels = H_space.num_levels
        self.ndim = H_space.ndim
        self.CP_status = {lev: np.zeros(H_space.fn_sh[lev]) for lev in range(H_space.num_levels)}
        self.CP_status[0] = np.ones_like(self.CP_status[0])
    
    def update_CP(self, ctrl_pts, fns, Coeffs):
        new_ctrl_pts = {lev: np.zeros_like(ctrl_pts[lev]) for lev in range(self.num_levels)}
        for lev in range(1, self.num_levels):
            # newly activated control points
            curr_coeff = Coeffs[lev-1]
            for CP in np.ndindex(fns[lev].shape):
                if self.CP_status[lev][CP]==1 and fns[lev][CP]==1:
                    new_ctrl_pts[lev][CP] = ctrl_pts[lev][CP]
                elif self.CP_status[lev][CP]==0 and fns[lev][CP]==1:
                    new_ctrl_pts[lev][CP] = [np.einsum(curr_coeff[dim].T[CP[dim]], ctrl_pts[lev]) for dim in range(self.ndim)]

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
