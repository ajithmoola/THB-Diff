import numpy as np
from funcs import refine_knotvector

class BSpline:

    def __init__(self, knotvector, degree):
        self.knotvector = knotvector
        self.degree = degree
    
    def refine_bspline(self):
        new_knotvector = refine_knotvector(self.knotvector)
        self.next_level = BSpline(new_knotvector, self.degree)
        return self.next_level


class TensorProduct:

    def __init__(self, args):
        self.bsplines = args
    
    def refine_tensorproduct(self):
        self.next_level = TensorProduct((bs.refine_bspline() for bs in self.bsplines))
        return self.next_level


class Hierarchy:

    def __init__(self, InitialSpace):
        self.H = {0: InitialSpace}
        self.degrees = (bs.degree for bs in InitialSpace.bsplines)
    
    def construct_multilevel_sequence(self, num_levels):
        self.num_levels = num_levels
        for lev in range(1, num_levels):
            self.H[lev] = self.H[lev-1].refine_tensorproduct()

    def add_level(self):
        refined_space = self.H[self.max_level]
        self.max_level += 1
        self.num_levels += 1
        self.H[self.max_level] = refined_space.refine_tensorproduct()


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
