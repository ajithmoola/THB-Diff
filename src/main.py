from space import Space
from THB_utils import *
from datastructures import *

kv1 = np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
p1 = 2
kv2 = np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1])
p2 = 3
bs1 = BSpline(kv1, p1)
bs2 = BSpline(kv2, p2)

tp = TensorProduct(bsplines=[bs1, bs2])

h_obj = Hierarchy(InitialSpace=tp)

h_obj.construct_multilevel_sequence(num_levels=3)

# print(h_obj.H[1].bsplines[0].knotvector)

h_space = Space(h_obj)
h_space.build_hierarchy_from_domain_sequence()
h_space.refine_cell((2, 2), 0)
print(h_space.cells[0][2, 2])
h_space.build_hierarchy_from_domain_sequence()
# print(np.nonzero(h_space.fns[1]))

ac_cells = compute_active_cells_active_supp(h_space.cells, h_space.fns, h_space.degrees)
print(ac_cells[1][(4, 4)])

fn_coeffs = compute_projection_matrices(h_space.fns, h_space.Coeff, h_space.degrees)
print(fn_coeffs[1][(0, 0)])