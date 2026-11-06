from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jit, jvp, vmap

from THBDiff.thbdiff2D import HierarchicalSplineSpace
from THBDiff.bspline_funcs import basis_fns


@partial(jit, static_argnums=(9,))
def evaluate_3D(
    params,
    ctrl_pts,
    coeffs,
    offsets,
    active_ids,
    spans,
    pt_ids,
    fn_ids,
    max_knotvectors,
    degrees,
):
    local_basis = [
        basis_fns(params[:, dim], max_knotvectors[dim], degrees[dim]) for dim in range(3)
    ]
    output = jnp.zeros((params.shape[0], ctrl_pts.shape[-1]))
    for c, o, ids, pt, fn in zip(coeffs, offsets, active_ids, pt_ids, fn_ids):
        rel = spans[pt] - o[fn]
        idx, weights = [], []
        for dim in range(3):
            i = rel[:, dim, None] + jnp.arange(degrees[dim] + 1)
            valid = (i >= 0) & (i < c.shape[dim + 1])
            idx.append(jnp.clip(i, 0, c.shape[dim + 1] - 1))
            weights.append(jnp.where(valid, local_basis[dim][pt], 0.0))
        vals = c[
            fn[:, None, None, None],
            idx[0][:, :, None, None],
            idx[1][:, None, :, None],
            idx[2][:, None, None, :],
        ]
        phi = jnp.einsum("mabc, ma, mb, mc -> m", vals, *weights)
        output = output.at[pt].add(phi[:, None] * ctrl_pts[ids[fn]])
    return output


@partial(jit, static_argnums=(9,))
def compute_thb_jacobian_3D(
    params,
    ctrl_pts,
    coeffs,
    offsets,
    active_ids,
    spans,
    pt_ids,
    fn_ids,
    max_knotvectors,
    degrees,
):
    def f(p):
        return evaluate_3D(
            p, ctrl_pts, coeffs, offsets, active_ids, spans, pt_ids, fn_ids, max_knotvectors, degrees
        )

    tangents = jnp.eye(params.shape[1])[:, None, :] * jnp.ones_like(params)
    cols = vmap(lambda t: jvp(f, (params,), (t,))[1])(tangents)
    return cols.transpose(1, 2, 0)


class HierarchicalSplineSpace3D(HierarchicalSplineSpace):

    def compute_refinement_operators(self):
        max_lev = self.max_lev
        first, last = {}, {}
        for lev in range(max_lev):
            nonzero = [np.abs(self.Coeff[lev][dim]) > 1e-12 for dim in range(self.ndim)]
            first[lev] = [nz.argmax(axis=1) for nz in nonzero]
            last[lev] = [nz.shape[1] - 1 - nz[:, ::-1].argmax(axis=1) for nz in nonzero]

        self.fn_offsets = np.concatenate(
            [[0], np.cumsum([np.prod(self.sh_fns[lev]) for lev in range(self.num_levels)])]
        ).astype(np.int64)
        self.compact_index = np.full(self.fn_offsets[-1], -1, dtype=np.int32)
        self.coeffs, self.offsets, self.active_ids = {}, {}, {}

        for lev in range(self.num_levels):
            act = np.argwhere(self.fns[lev] == 1)
            self.active_ids[lev] = (
                self.fn_offsets[lev] + np.ravel_multi_index(act.T, self.sh_fns[lev])
            ).astype(np.int32)
            self.compact_index[self.active_ids[lev]] = np.arange(len(act))

            off = act
            width = np.ones(self.ndim, dtype=np.int64)
            c = np.ones((len(act), 1, 1, 1))
            if len(act) > 0:
                for k in range(lev, max_lev):
                    new_off = np.stack(
                        [first[k][d][off[:, d]] for d in range(self.ndim)], axis=1
                    )
                    end = np.stack(
                        [
                            last[k][d][np.minimum(off[:, d] + width[d] - 1, self.sh_fns[k][d] - 1)]
                            for d in range(self.ndim)
                        ],
                        axis=1,
                    )
                    new_width = (end - new_off + 1).max(axis=0)

                    blocks = []
                    for d in range(self.ndim):
                        C = np.pad(self.Coeff[k][d], ((0, width[d]), (0, new_width[d])))
                        rows = off[:, d, None, None] + np.arange(width[d])[None, :, None]
                        cols = new_off[:, d, None, None] + np.arange(new_width[d])[None, None, :]
                        blocks.append(C[rows, cols])
                    c = np.einsum("nabc, nad, nbe, ncf -> ndef", c, *blocks, optimize=True)

                    fns_pad = np.pad(self.fns[k + 1], [(0, w) for w in new_width])
                    grids = tuple(
                        new_off[:, d].reshape(-1, 1, 1, 1)
                        + np.arange(new_width[d]).reshape([-1 if i == d + 1 else 1 for i in range(4)])
                        for d in range(self.ndim)
                    )
                    c[fns_pad[grids] == 1] = 0
                    off, width = new_off, new_width

            self.coeffs[lev] = c.astype(np.float32)
            self.offsets[lev] = off.reshape(-1, self.ndim).astype(np.int32)

    def update_CP(self, ctrl_pts):
        if not isinstance(ctrl_pts, dict):
            ctrl_pts = self.CP_arr_to_dict(np.array(ctrl_pts))
        else:
            ctrl_pts = self.CP_arr_to_dict(np.array(self.CP_dict_to_arr(ctrl_pts)))
        CP_status = self.refinement_history[-2]
        for lev in range(1, self.num_levels):
            C = self.Coeff[lev - 1]
            refined = np.einsum(
                "ai, bj, ck, abcd -> ijkd", C[0], C[1], C[2], ctrl_pts[lev - 1], optimize=True
            )
            new = (CP_status[lev] == 0) & (self.fns[lev] == 1)
            ctrl_pts[lev][new] = refined[new]

        return self.CP_arr_to_dict(jnp.array(self.CP_dict_to_arr(ctrl_pts)))
