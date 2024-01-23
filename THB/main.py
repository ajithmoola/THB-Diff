from space import Space
from THB_core import *
from datastructures import *
from funcs import *


bs1 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)
bs2 = BSpline(knotvector=np.array([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1]), degree=3)
bs3 = BSpline(knotvector=np.array([0, 0, 0, 0.33, 0.66, 0.8, 1, 1, 1]), degree=2)

tp = TensorProduct(bsplines=[bs1, bs2, bs3])

# h_obj = Hierarchy(InitialSpace=tp)
# h_obj.construct_multilevel_sequence(num_levels=3)

# print(h_obj.H[1].bsplines[0].knotvector)

h_space = Space(tensor_product=tp, num_levels=3)
h_space.build_hierarchy_from_domain_sequence()
# h_space._refine_cell((2, 2), 0)
h_space.refine_basis_fn((2, 2, 2), 0)
h_space.build_hierarchy_from_domain_sequence()
print(h_space.fns[0][2, 2, 2])
# print(np.nonzero(h_space.fns[1]))

ac_cells_ac_supp = compute_active_cells_active_supp(h_space.cells, h_space.fns, h_space.degrees)
# print(ac_cells_ac_supp[1][(4, 4)])

fn_coeffs = compute_fn_projection_matrices(h_space.fns, h_space.Coeff, h_space.degrees)
# print(fn_coeffs[0][(6, 6)][0])

param_pts = generate_parametric_coordinates((100, 100, 100))
# print(param_pts.shape)

ac_spans = compute_active_span(param_pts, h_space.knotvectors, h_space.cells, h_space.degrees, h_space.sh_fns)
# print(len(ac_spans))

compute_tensor_product(param_pts, ac_spans, ac_cells_ac_supp, fn_coeffs, h_space.sh_fns, h_space.knotvectors, h_space.degrees)