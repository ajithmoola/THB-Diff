from jax import value_and_grad, grad, jit
import jax.numpy as jnp
from THB.core import *
from THB.jax import *


class THB_layer:

    def __init__(self, h_space):
        self.h_space = h_space

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
            self.h_space.sh_fns,
        )

        PHI, num_supp = compute_basis_fns_tp_parallel(
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

    def evaluate(self):
        return Evaluate_JAX(self.CP, self.Jm, self.PHI, self.segment_ids, self.num_pts)

    def loss_and_grad(self, loss_fn):
        return jit(value_and_grad(loss_fn, argnums=0), static_argnums=(4,))

    @staticmethod
    def fitting_MSE_loss(CP, Jm, PHI, segment_ids, num_pts, target):
        output = Evaluate_JAX(CP, Jm, PHI, segment_ids, num_pts)
        loss = jnp.mean((output - target) ** 2)
        return loss

    @staticmethod
    def fitting_CD_loss(CP, Jm, PHI, segment_ids, num_pts, target):
        output = Evaluate_JAX(CP, Jm, PHI, segment_ids, num_pts)

        norm1 = jnp.sum(output**2, axis=1, keepdims=True)
        norm2 = jnp.sum(target**2, axis=1, keepdims=True)

        dists = norm1 + norm2.T - 2 * jnp.einsum("ij, kj -> ik", norm1, norm2)

        min_dist_1_to_2 = jnp.min(dists, axis=1)
        min_dist_2_to_1 = jnp.min(dists, axis=0)

        CD = jnp.sum(min_dist_1_to_2) + jnp.sum(min_dist_2_to_1)
        return CD
