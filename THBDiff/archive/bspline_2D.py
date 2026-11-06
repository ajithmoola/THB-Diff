from functools import partial

import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import jacfwd

from dbspline.bspline_core import basis_fns, find_span_array, basis_fns_full


@partial(jit, static_argnums=(2,))
def compute_basis_at_point(param, knotvectors, degrees):
    b_fns = [
        basis_fns(param[dim : dim + 1], knotvectors[dim], degrees[dim]).squeeze()
        for dim in range(2)
    ]
    tp = jnp.outer(b_fns[0], b_fns[1])
    return tp


@partial(jit, static_argnums=(2))
def compute_tensor_product(params, knotvectors, degrees):
    basis_fns = vmap(compute_basis_at_point, in_axes=(0, None, None))(
        params, knotvectors, degrees
    )
    return basis_fns


@partial(jit, static_argnums=(2,))
def compute_basis_at_point_full(param, knotvectors, degrees):
    b_fns = [
        basis_fns_full(param[dim : dim + 1], knotvectors[dim], degrees[dim]).squeeze()
        for dim in range(2)
    ]
    tp = jnp.outer(b_fns[0], b_fns[1])
    return tp


@partial(jit, static_argnums=(2))
def compute_tensor_product_full(params, knotvectors, degrees):
    basis_fns = vmap(compute_basis_at_point_full, in_axes=(0, None, None))(
        params, knotvectors, degrees
    )
    return basis_fns


def evaluate_at_point(tp, ctrl_pts, ctrl_pts_indices):
    pts = jnp.einsum(
        "ijk, ij -> k",
        ctrl_pts[ctrl_pts_indices[0], ctrl_pts_indices[1]].squeeze(),
        tp,
    )
    return pts


def evaluate(tp, ctrl_pts, CP_indices):
    pts = vmap(evaluate_at_point, in_axes=(0, None, 0))(tp, ctrl_pts, CP_indices)
    return pts


@partial(jit, static_argnums=(3,))
def compute_tp_and_evaluate_at_point(param, ctrl_pts, knotvectors, degrees, CP_indices):
    tp = compute_basis_at_point(param, knotvectors, degrees)
    pts = evaluate_at_point(tp, ctrl_pts, CP_indices)
    return pts


@partial(jit, static_argnums=(3))
def compute_tp_and_evaluate(params, ctrl_pts, knotvectors, degrees, CP_indices):
    pts = vmap(compute_tp_and_evaluate_at_point, in_axes=(0, None, None, None, 0))(
        params, ctrl_pts, knotvectors, degrees, CP_indices
    )
    return pts


@partial(jit, static_argnums=(3,))
def bspline_jacobian_at_point(param, ctrl_pts, knotvectors, degrees, CP_indices):
    jac = jacfwd(compute_tp_and_evaluate_at_point, argnums=0)(
        param, ctrl_pts, knotvectors, degrees, CP_indices
    )
    return jac


@partial(jit, static_argnums=(3))
def bspline_jacobian(params, ctrl_pts, knotvectors, degrees, CP_indices):
    jac = vmap(bspline_jacobian_at_point, in_axes=(0, None, None, None, 0))(
        params, ctrl_pts, knotvectors, degrees, CP_indices
    )
    return jac


def get_CP_indices(knotvectors, degrees, params):
    p, q = degrees[0], degrees[1]
    kv1, kv2 = knotvectors[0], knotvectors[1]
    spans = jnp.stack(
        [find_span_array(params[:, 0], kv1, p), find_span_array(params[:, 1], kv2, q)]
    ).T
    x_offsets, y_offsets = jnp.meshgrid(
        jnp.arange(p + 1), jnp.arange(q + 1), indexing="ij"
    )
    x_indices = spans[:, 0, jnp.newaxis, jnp.newaxis] + x_offsets - p
    y_indices = spans[:, 1, jnp.newaxis, jnp.newaxis] + y_offsets - q
    CP_indices = [x_indices, y_indices]
    return jnp.array(CP_indices).transpose(1, 0, 2, 3)


if __name__ == "__main__":
    import jax
    from bspline_funcs import generate_parametric_coordinates

    # from plot import plot_valve_wireframe

    p, q = 2, 2
    kv1 = jnp.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
    kv2 = jnp.array([0, 0, 0, 0.33, 0.66, 1, 1, 1])

    knotvectors = (kv1, kv2)
    degrees = (p, q)

    fn_sh = [len(kv) - 1 - degree for kv, degree in zip(knotvectors, degrees)]

    key = jax.random.PRNGKey(0)
    CP = jax.random.uniform(key, shape=(*fn_sh, 2))

    def rect_control_points(n, m, width=1.0, height=1.0, center=(0, 0)):
        u = jnp.linspace(0, 1, n)
        v = jnp.linspace(0, 1, m)
        u_grid, v_grid = jnp.meshgrid(u, v, indexing="ij")

        x = center[0] - width / 2 + width * u_grid
        y = center[1] - height / 2 + height * v_grid

        return jnp.stack([x, y], axis=-1)

    CP = rect_control_points(*fn_sh, width=1.0, height=1.0, center=(0, 0))
    CP = jnp.dstack([CP, jnp.zeros((7, 5, 1))])

    print(CP.shape)

    params = generate_parametric_coordinates((500, 500))

    CP_indices = get_CP_indices(knotvectors, degrees, params)

    tp = compute_tensor_product(params, knotvectors, degrees)

    pts = compute_tp_and_evaluate(params, CP, knotvectors, degrees, CP_indices)

    # jac = bspline_jacobian(params, CP, knotvectors, degrees, CP_indices)
    # print(tp.shape, CP_indices.shape, pts.shape, jac.shape)

    # plot_valve_wireframe(pts, (500, 500), color="green", linewidth=0.5, dpi=150)
