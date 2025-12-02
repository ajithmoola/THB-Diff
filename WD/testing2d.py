from THBDiff.thbdiff2D import BSpline, HierarchicalSplineSpace, TensorProduct
import numpy as np

if __name__ == "__main__":
    bs1 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)
    bs2 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)

    tp = TensorProduct([bs1, bs2])
    h_space = HierarchicalSplineSpace(tp, num_levels=3)
    h_space.build_hierarchy_from_domain_sequence()
    h_space.plot_adaptive_grid_2D("thb_diff_lvl0.png")
    h_space._refine_basis_fn((5, 5), 0)
    h_space._refine_basis_fn((2, 2), 0)
    h_space.build_hierarchy_from_domain_sequence()
    h_space.plot_adaptive_grid_2D("thb_diff_lvl1.png")
    h_space._refine_basis_fn((2, 2), 1)
    h_space._refine_basis_fn((10, 10), 1)
    h_space.build_hierarchy_from_domain_sequence()
    h_space.plot_adaptive_grid_2D("thb_diff_lvl2.png")
