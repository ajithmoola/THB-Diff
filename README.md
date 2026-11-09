# THBDiff

Differentiable truncated hierarchical B-splines (THB-splines) in JAX, for 2D and 3D.

## Install

```bash
pip install -e .
```

This also compiles the `tmatrix_cpp` C++ extension (pybind11) used to assemble knot-insertion matrices. If the extension is not available, a Python fallback is used.

## Usage

```python
import numpy as np
import jax.numpy as jnp
from THBDiff.thb import BSpline, TensorProduct
from THBDiff.thbdiff3D import HierarchicalSplineSpace3D, evaluate_3D, compute_thb_jacobian_3D
from THBDiff.bspline_funcs import generate_parametric_coordinates

kv = np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
tp = TensorProduct([BSpline(knotvector=kv, degree=2) for _ in range(3)])
h_space = HierarchicalSplineSpace3D(tp, num_levels=3)

h_space.build_hierarchy_from_domain_sequence()
h_space._refine_basis_fn((5, 5, 5), 0)
h_space.build_hierarchy_from_domain_sequence()

params = generate_parametric_coordinates((50, 50, 50))
ctrl_pts = jnp.array(h_space.CP_dict_to_arr(h_space.GA))
coeffs, offsets, active_ids = h_space.get_compact_coeffs()
spans, pt_ids, fn_ids = h_space.prepare_evaluation(params)
args = (params, ctrl_pts, coeffs, offsets, active_ids, spans, pt_ids, fn_ids,
        h_space.knotvectors[h_space.max_lev], h_space.degrees)

pts = evaluate_3D(*args)
jac = compute_thb_jacobian_3D(*args)
```

For 2D, use `HierarchicalSplineSpace`, `evaluate` and `compute_thb_jacobian` from `THBDiff.thbdiff2D`.

## Layout

- `THBDiff/thb.py`: base `HierarchicalSplineSpace`, `BSpline`, `TensorProduct`
- `THBDiff/thbdiff2D.py`, `THBDiff/thbdiff3D.py`: 2D and 3D spaces, evaluation and Jacobians
- `THBDiff/bspline_funcs.py`: B-spline basis functions, knot refinement, Greville points
- `THBDiff/tmatrix_cpp.cpp`: C++ knot-insertion matrix assembly
- `WD/testing3d.py`: 3D refinement example that writes the adaptive grid as a PLY wireframe
