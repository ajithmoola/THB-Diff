import mlx.core as mx
import numpy as np


def prepare_data_for_evaluation_mlx(
    PHI, ac_spans, num_supp, ctrl_pts, ac_cells_ac_supp, fn_sh
):
    max_lev = max(ctrl_pts.keys())
    nCP = np.zeros(max_lev + 2, dtype=np.int_)
    CP_dim = ctrl_pts[0].shape[-1]

    segment_lengths = np.array(num_supp)
    num_pts = segment_lengths.size
    segment_ids = mx.array(np.repeat(np.arange(num_pts), segment_lengths))

    PHI = mx.array(PHI).reshape(-1, 1)

    for lev in range(1, max_lev + 2):
        nCP[lev] = nCP[lev - 1] + np.prod(fn_sh[lev - 1])

    ctrl_pts_flat = [
        np.array(ctrl_pts[lev]).reshape(-1, CP_dim).astype(np.float32)
        for lev in range(max_lev + 1)
    ]
    ctrl_pts = mx.array(np.vstack(ctrl_pts_flat))

    Jm = [
        nCP[fn_lev] + np.ravel_multi_index(fnIdx, fn_sh[fn_lev])
        for cell_lev, cellIdx in ac_spans
        for fn_lev, fnIdx in ac_cells_ac_supp[cell_lev][cellIdx]
    ]

    Jm = mx.array(np.array(Jm).astype(np.int32))

    return ctrl_pts, Jm, PHI, segment_ids, num_pts


def Evaluate_MLX(ctrl_pts, Jm, PHI, segment_ids, num_pts):
    prod = PHI * ctrl_pts[Jm]
    output = mx.zeros((num_pts, ctrl_pts.shape[1])).at[segment_ids].add(prod)
    return output
