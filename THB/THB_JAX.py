import jax.numpy as jnp
from jax import lax, jit
import numpy as np

def prepare_data_for_evaluation_jax(PHI, ac_spans, num_supp, ctrl_pts, ac_cells_ac_supp, fn_sh):
    max_lev = max(ctrl_pts.keys())
    nCP = np.zeros(max_lev+2, dtype=np.int_)
    CP_dim = ctrl_pts[0].shape[-1]

    num_supp = jnp.array(num_supp)
    segment_lengths = num_supp
    num_pts = segment_lengths.size
    segment_ids = jnp.repeat(jnp.arange(num_pts), segment_lengths)

    PHI = jnp.array(PHI).astype(jnp.float32).reshape(-1, 1)
    
    for lev in range(1, max_lev+2):
        nCP[lev] = nCP[lev-1] + np.prod(fn_sh[lev-1])
    
    ctrl_pts_flat = [jnp.array(ctrl_pts[lev]).reshape(-1, CP_dim).astype(jnp.float32) for lev in range(max_lev+1)]
    ctrl_pts = jnp.vstack(ctrl_pts_flat)

    Jm = [nCP[fn_lev] + np.ravel_multi_index(fnIdx, fn_sh[fn_lev]) for cell_lev, cellIdx in ac_spans for fn_lev, fnIdx in ac_cells_ac_supp[cell_lev][cellIdx]]
    
    Jm = jnp.array(Jm)

    return ctrl_pts, Jm, PHI, segment_ids, num_pts

def Evaluate_JAX(ctrl_pts, Jm, PHI, segment_ids, num_pts):
    output = jnp.zeros((num_pts, 3)).at[segment_ids].add(ctrl_pts[Jm] * PHI)
    return output