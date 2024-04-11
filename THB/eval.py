from jax import value_and_grad, grad, jit
import jax.numpy as jnp
from THB.core import *
from THB.funcs import grevilleAbscissae
from THB.jax import *
import numpy as np


class THB_layer:

    def __init__(self, h_space):
        self.h_space = h_space

    def initialize_control_points(self):
        self.GA = {
            lev: np.zeros((*self.h_space.sh_fns[lev], self.h_space.ndim))
            for lev in range(self.h_space.num_levels)
        }
        self.GA[0] = grevilleAbscissae(
            self.h_space.sh_fns[0], self.h_space.degrees, self.h_space.knotvectors[0]
        )
        self.CP_status = {
            lev: np.zeros_like(self.h_space.fns[lev])
            for lev in range(self.h_space.num_levels)
        }
        self.CP_status[0] = np.ones_like(self.CP_status[0])

    def update_GA(self):
        self.GA = self.refine_ctrl_pts(self.GA)

    @staticmethod
    def CP_arr_to_dict(CP_arr, sh_fns, num_levels):
        CP_arr = np.array(CP_arr)
        nCP = np.array(
            [0] + [np.prod(sh_fns[lev]) for lev in range(num_levels)]
        ).cumsum()
        ctrl_pts = {
            lev: CP_arr[nCP[lev] : nCP[lev + 1]].reshape(*sh_fns[lev], CP_arr.shape[1])
            for lev in range(num_levels)
        }
        return ctrl_pts

    def update_ctrl_pts(self, CP_arr):
        if type(CP_arr) == dict:
            new_ctrl_pts = CP_arr
        else:
            # nCP = [0] + [
            #     np.prod(
            #         self.h_space.sh_fns[lev] for lev in range(self.h_space.num_levels)
            #     )
            # ]
            # ctrl_pts = {
            #     lev: CP_arr[nCP[lev] : nCP[lev + 1]].reshape(*self.h_space.sh_fns[lev])
            #     for lev in range(self.h_space.num_levels)
            # }
            ctrl_pts = self.CP_arr_to_dict(
                CP_arr, self.h_space.sh_fns, self.h_space.num_levels
            )

        # Refining control points
        new_ctrl_pts = self.refine_ctrl_pts(ctrl_pts)
        self.GA = self.refine_ctrl_pts(self.GA)

        # TODO: Coarsening control points
        self.CP_status = deepcopy(self.h_space.fns)
        return new_ctrl_pts

    def refine_ctrl_pts(self, ctrl_pts):
        for lev in range(1, self.h_space.num_levels):
            curr_coeff = self.h_space.Coeff[lev - 1]
            for CP in np.ndindex(self.h_space.fns[lev].shape):
                if self.CP_status[lev][CP] == 0 and self.h_space.fns[lev][CP] == 1:
                    cp_coeff = [
                        curr_coeff[dim].T[CP[dim]] for dim in range(self.h_space.ndim)
                    ]
                    tp = compute_tensor_product(cp_coeff)
                    ctrl_pts[lev][CP] = np.sum(
                        tp[..., np.newaxis] * ctrl_pts[lev - 1],
                        axis=tuple(range(len(tp.shape))),
                    )
        return ctrl_pts

    def compute_refinement_operators(self):
        self.h_space.build_hierarchy_from_domain_sequence()
        self.ac_cells = compute_active_cells_active_supp(
            self.h_space.cells, self.h_space.fns, self.h_space.degrees
        )
        self.fn_coeffs = compute_refinement_operators(
            self.h_space.fns, self.h_space.Coeff, self.h_space.degrees
        )

    def compute_tensor_product(self, parameters, ctrl_pts):
        self.ac_spans = compute_active_span(
            parameters,
            self.h_space.knotvectors,
            self.h_space.cells,
            self.h_space.degrees,
        )

        # self.ac_spans, num_supp, sub_coeffs = faster_compute_active_span(
        #     parameters,
        #     self.h_space.knotvectors,
        #     self.h_space.cells,
        #     self.h_space.degrees,
        #     self.h_space.sh_fns,
        #     self.ac_cells,
        #     self.fn_coeffs,
        # )

        # basis_fns = compute_basis_fns(
        #     parameters,
        #     self.h_space.knotvectors,
        #     self.h_space.degrees,
        #     self.h_space.ndim,
        #     self.h_space.num_levels - 1,
        # )

        # PHI = compute_basis_fns_tp_vectorized(
        #     self.h_space.degrees,
        #     num_supp,
        #     sub_coeffs,
        #     basis_fns,
        # )
        PHI, num_supp = compute_THB_fns_tp_parallel(
            parameters,
            self.ac_spans,
            self.ac_cells,
            self.fn_coeffs,
            self.h_space.sh_fns,
            self.h_space.knotvectors,
            self.h_space.degrees,
        )

        self.CP, self.Jm, self.PHI, self.segment_ids, self.num_pts = (
            prepare_data_for_evaluation_jax(
                PHI,
                self.ac_spans,
                num_supp,
                ctrl_pts,
                self.ac_cells,
                self.h_space.sh_fns,
            )
        )

    def refine_cells(self, cells):
        for lev, cellIdx in cells:
            self.h_space._refine_cell(cellIdx, lev)

    def evaluate(self):
        return Evaluate_JAX(self.CP, self.Jm, self.PHI, self.segment_ids, self.num_pts)

    def output_and_CP_grad(self, obj_fn):
        return jit(value_and_grad(obj_fn, argnums=0), static_argnums=4)

    @staticmethod
    def fitting_MSE_loss(CP, Jm, PHI, segment_ids, num_pts, target):
        output = Evaluate_JAX(CP, Jm, PHI, segment_ids, num_pts)
        loss = jnp.mean((output - target) ** 2)
        return loss

    @staticmethod
    def fitting_CD_loss(CP, Jm, PHI, segment_ids, out_shape, target):
        output = Evaluate_JAX(CP, Jm, PHI, segment_ids, out_shape)

        norm1 = jnp.sum(output**2, axis=1, keepdims=True)
        norm2 = jnp.sum(target**2, axis=1, keepdims=True)

        dists = norm1 + norm2.T - 2 * jnp.einsum("ij, kj -> ik", norm1, norm2)

        min_dist_1_to_2 = jnp.min(dists, axis=1)
        min_dist_2_to_1 = jnp.min(dists, axis=0)

        CD = jnp.sum(min_dist_1_to_2) + jnp.sum(min_dist_2_to_1)
        return CD


class PDE:

    def __init__(self, PDE):
        self.PDE = PDE
