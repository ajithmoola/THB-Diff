from THBDiff.thb import BSpline, TensorProduct
from THBDiff.thbdiff3D import (
    HierarchicalSplineSpace3D,
    evaluate_3D,
    compute_thb_jacobian_3D,
)
from THBDiff.bspline_funcs import generate_parametric_coordinates
import numpy as np
import jax.numpy as jnp

bs1 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)
bs2 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)
bs3 = BSpline(knotvector=np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]), degree=2)

tp = TensorProduct([bs1, bs2, bs3])
h_space = HierarchicalSplineSpace3D(tp, num_levels=3)
params = generate_parametric_coordinates((50, 50, 50))

h_space.build_hierarchy_from_domain_sequence()
h_space._refine_basis_fn((5, 5, 5), 0)
h_space._refine_basis_fn((2, 2, 2), 0)
h_space.build_hierarchy_from_domain_sequence()
h_space._refine_basis_fn((2, 2, 2), 1)
h_space._refine_basis_fn((10, 10, 10), 1)
h_space.build_hierarchy_from_domain_sequence()

ctrl_pts = jnp.array(h_space.CP_dict_to_arr(h_space.GA))
coeffs, offsets, active_ids = h_space.get_compact_coeffs()
spans, pt_ids, fn_ids = h_space.prepare_evaluation(params)
max_kv = h_space.knotvectors[h_space.max_lev]

pts = evaluate_3D(
    params,
    ctrl_pts,
    coeffs,
    offsets,
    active_ids,
    spans,
    pt_ids,
    fn_ids,
    max_kv,
    h_space.degrees,
)
jac = compute_thb_jacobian_3D(
    params,
    ctrl_pts,
    coeffs,
    offsets,
    active_ids,
    spans,
    pt_ids,
    fn_ids,
    max_kv,
    h_space.degrees,
)

print([int(h_space.fns[lev].sum()) for lev in range(h_space.num_levels)])
print(jnp.abs(pts - params).max())
print(jnp.abs(jac - jnp.eye(3)).max())

corners = []
for lev in range(h_space.num_levels):
    knots = [np.unique(h_space.knotvectors[lev][dim]) for dim in range(3)]
    for cell in h_space.ac_cells[lev].keys():
        lo = [knots[dim][cell[dim]] for dim in range(3)]
        hi = [knots[dim][cell[dim] + 1] for dim in range(3)]
        corners.append(
            [[x, y, z] for z in (lo[2], hi[2]) for y in (lo[1], hi[1]) for x in (lo[0], hi[0])]
        )
corner_params = jnp.array(
    np.minimum(np.array(corners).reshape(-1, 3), 1 - 1e-7), dtype=jnp.float32
)

spans, pt_ids, fn_ids = h_space.prepare_evaluation(corner_params)
grid_pts = evaluate_3D(
    corner_params,
    ctrl_pts,
    coeffs,
    offsets,
    active_ids,
    spans,
    pt_ids,
    fn_ids,
    max_kv,
    h_space.degrees,
)

hex_edges = np.array(
    [
        [0, 1], [2, 3], [4, 5], [6, 7],
        [0, 2], [1, 3], [4, 6], [5, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
)
edges = np.unique(
    np.sort((np.arange(len(corners))[:, None, None] * 8 + hex_edges[None]).reshape(-1, 2), axis=1),
    axis=0,
)
vertices = np.asarray(grid_pts)
with open("thb_diff_3d.ply", "w") as f:
    f.write(
        f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element edge {len(edges)}\nproperty int vertex1\nproperty int vertex2\nend_header\n"
    )
    np.savetxt(f, vertices, fmt="%.7f")
    np.savetxt(f, edges, fmt="%d")

print(len(corners))
print(jnp.abs(grid_pts - corner_params).max())
