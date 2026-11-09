from itertools import product
from copy import deepcopy
from typing import Dict, Tuple, List
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import THBDiff.tmatrix_cpp as tmatrix_cpp

    USE_CPP_TMATRIX = True
except ImportError:
    USE_CPP_TMATRIX = False


def refine_knotvector(knotvector, p):
    # knots = jnp.unique(knotvector)
    knotvector = np.asarray(knotvector, dtype=np.float64)
    valid_knots = knotvector[p:-p]
    mids = 0.5 * (valid_knots[1:] + valid_knots[:-1])
    refined_knotvector = np.concatenate(
        [
            knotvector[:p],
            np.unique(np.sort(np.concatenate([valid_knots, mids]))),
            knotvector[-p:],
        ]
    )
    return refined_knotvector


def generate_parametric_coordinates(shape, endpoint=False, delta=1e-5):
    ndim = len(shape)
    pts = jnp.hstack(
        tuple(
            map(
                lambda x: x.reshape(-1, 1),
                jnp.meshgrid(
                    *[
                        jnp.linspace(0, 1 - delta, shape[dim], endpoint=endpoint)
                        for dim in range(ndim)
                    ]
                ),
            )
        )
    )
    return pts


def grevilleAbscissae(fn_sh, degrees, knotvectors):
    ndim = len(fn_sh)
    CP = np.zeros((*fn_sh, ndim))

    for pt in np.ndindex(fn_sh):
        CP[pt] = jnp.array(
            [
                np.sum(knotvectors[dim][pt[dim] + 1 : pt[dim] + degrees[dim] + 1])
                / degrees[dim]
                for dim in range(ndim)
            ]
        )

    return jnp.array(CP)


@jit
def find_span_array(params, U, degree):
    n = len(U) - degree - 1
    indices = jnp.searchsorted(U, params, side="right") - 1
    indices = jnp.where(indices > n, n, indices)
    indices = jnp.where(params == U[n + 1], n, indices)
    return indices


@jit
def divisionbyzero(numerator, denominator):
    force_zero = jnp.logical_and(numerator == 0, denominator == 0)
    return jnp.where(force_zero, jnp.float32(0.0), numerator) / jnp.where(
        force_zero, jnp.float32(1.0), denominator
    )


def basis_fns(params, knotvector, degree):
    params1d = jnp.array(params)
    U = jnp.expand_dims(params1d, -1)
    knots = jnp.expand_dims(knotvector, 0)
    spans = find_span_array(params1d, knotvector, degree)
    K = jnp.where(
        knots == knotvector[-1], knotvector[-1] + jnp.finfo(U.dtype).eps, knots
    )
    t1 = U >= K[..., :-1]
    t2 = U < K[..., 1:]
    N = (t1 * t2) + 0.0
    for p in range(1, degree + 1):
        term1 = divisionbyzero(
            N[..., :-1] * (U - K[..., : -p - 1]), K[..., p:-1] - K[..., : -p - 1]
        )
        term2 = divisionbyzero(
            N[..., 1:] * (K[..., p + 1 :] - U), K[..., p + 1 :] - K[..., 1:-p]
        )
        N = term1 + term2

    span_ind = jnp.arange(degree + 1) + spans[:, jnp.newaxis] - degree
    row_indices = jnp.arange(span_ind.shape[0])[:, jnp.newaxis]
    non_zero_N = N[row_indices, span_ind]
    return non_zero_N


def basis_fns_full(params, knotvector, degree):
    params1d = jnp.array(params)
    U = jnp.expand_dims(params1d, -1)
    knots = jnp.expand_dims(knotvector, 0)
    K = jnp.where(
        knots == knotvector[-1], knotvector[-1] + jnp.finfo(U.dtype).eps, knots
    )
    t1 = U >= K[..., :-1]
    t2 = U < K[..., 1:]
    N = (t1 * t2) + 0.0
    for p in range(1, degree + 1):
        term1 = divisionbyzero(
            N[..., :-1] * (U - K[..., : -p - 1]), K[..., p:-1] - K[..., : -p - 1]
        )
        term2 = divisionbyzero(
            N[..., 1:] * (K[..., p + 1 :] - U), K[..., p + 1 :] - K[..., 1:-p]
        )
        N = term1 + term2
    return N


def assemble_Tmatrix(knotVec, newKnotVec, knotVec_len, newKnotVec_len, p):
    if USE_CPP_TMATRIX:
        return tmatrix_cpp.assemble_Tmatrix_cpp(
            np.asarray(knotVec, dtype=np.float64),
            np.asarray(newKnotVec, dtype=np.float64),
            knotVec_len,
            newKnotVec_len,
            p,
        )
    else:
        # Fallback to Python implementation
        T1 = np.zeros((newKnotVec_len - 1, knotVec_len - 1))

        for i in range(newKnotVec_len - 1):
            for j in range(knotVec_len - 1):
                if (newKnotVec[i] >= knotVec[j]) and (newKnotVec[i] < knotVec[j + 1]):
                    T1[i, j] = 1

        for q in range(1, p + 1):
            T2 = np.zeros(((newKnotVec_len - q - 1), (knotVec_len - q - 1)))
            for i in range(newKnotVec_len - q - 1):
                for j in range(knotVec_len - q - 1):
                    if (knotVec[j + q] - knotVec[j] == 0) and (
                        knotVec[j + q + 1] - knotVec[j + 1] != 0
                    ):
                        T2[i, j] = (
                            (knotVec[j + q + 1] - newKnotVec[i + q])
                            / (knotVec[j + q + 1] - knotVec[j + 1])
                            * T1[i, j + 1]
                        )
                    if (knotVec[j + q] - knotVec[j] != 0) and (
                        knotVec[j + q + 1] - knotVec[j + 1] == 0
                    ):
                        T2[i, j] = (
                            (newKnotVec[i + q] - knotVec[j])
                            / (knotVec[j + q] - knotVec[j])
                            * T1[i, j]
                        )
                    if (knotVec[j + q] - knotVec[j] != 0) and (
                        knotVec[j + q + 1] - knotVec[j + 1] != 0
                    ):
                        T2[i, j] = (newKnotVec[i + q] - knotVec[j]) / (
                            knotVec[j + q] - knotVec[j]
                        ) * T1[i, j] + (knotVec[j + q + 1] - newKnotVec[i + q]) / (
                            knotVec[j + q + 1] - knotVec[j + 1]
                        ) * T1[
                            i, j + 1
                        ]

            T1 = T2
        return T1


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
