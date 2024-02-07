import torch
import THB_eval
import numpy as np
from THB.THB_utils import timer

class THBEval(torch.autograd.Function):
    @timer
    @staticmethod
    def forward(ctx, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum, device):
        ctx.save_for_backward(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        ctx.device = device
        if device=='cuda':
            return THB_eval.forward(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        else:
            return THB_eval.cpp_forward(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)

    @staticmethod
    def backward(ctx, grad_output):
        ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum = ctx.saved_tensors
        device = ctx.device
        if device=='cuda':
            grad_ctrl_pts = THB_eval.backward(grad_output, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        else:
            grad_ctrl_pts = THB_eval.cpp_backward(grad_output, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum)
        return grad_ctrl_pts, None, None, None, None
    

def prepare_data_for_CUDA_evaluation(PHI, ac_spans, num_supp, ctrl_pts, ac_cells_ac_supp, fn_sh, device):
    max_lev = max(ctrl_pts.keys())
    nCP = np.zeros(max_lev+2, dtype=np.int_)
    num_supp_cumsum = torch.from_numpy(np.concatenate([np.array([0]), num_supp]).cumsum()).to(device=device)
    PHI = torch.from_numpy(PHI).float().to(device=device)
    CP_dim = ctrl_pts[0].shape[-1]
    for lev in range(1, max_lev+2):
        nCP[lev] = nCP[lev-1] + np.prod(fn_sh[lev-1])
    
    ctrl_pts = torch.vstack([torch.from_numpy(ctrl_pts[lev]).reshape(-1, CP_dim).float() for lev in range(max_lev+1)]).to(device=device)
    
    Jm = [nCP[fn_lev] + np.ravel_multi_index(fnIdx, fn_sh[fn_lev]) for cell_lev, cellIdx in ac_spans for fn_lev, fnIdx in ac_cells_ac_supp[cell_lev][cellIdx]]

    Jm = torch.tensor(Jm).to(device)

    return ctrl_pts, Jm, PHI, num_supp_cumsum, device

def prepare_data_for_evaluation(PHI, ac_spans, num_supp, ctrl_pts, ac_cells_ac_supp, fn_sh, device):
    max_lev = max(ctrl_pts.keys())
    nCP = np.zeros(max_lev+2, dtype=np.int_)

    num_supp = torch.from_numpy(num_supp).to(device=device)
    segment_lengths = num_supp
    num_pts = segment_lengths.size(0)
    segment_ids = torch.repeat_interleave(torch.arange(num_pts, device=device), segment_lengths).unsqueeze(1).expand(-1, 3)

    PHI = torch.from_numpy(PHI).float().to(device=device).unsqueeze(1)
    CP_dim = ctrl_pts[0].shape[-1]

    for lev in range(1, max_lev+2):
        nCP[lev] = nCP[lev-1] + np.prod(fn_sh[lev-1])
    
    ctrl_pts = torch.vstack([torch.from_numpy(ctrl_pts[lev]).reshape(-1, CP_dim).float() for lev in range(max_lev+1)]).to(device=device)
    
    Jm = [nCP[fn_lev] + np.ravel_multi_index(fnIdx, fn_sh[fn_lev]) for cell_lev, cellIdx in ac_spans for fn_lev, fnIdx in ac_cells_ac_supp[cell_lev][cellIdx]]

    Jm = torch.tensor(Jm).to(device)

    return ctrl_pts, Jm, PHI, segment_ids, num_pts

@timer
@torch.compile
def Evaluate(ctrl_pts, Jm, PHI, segment_ids, num_pts):
    return torch.zeros((num_pts, 3)).to(device='cuda').scatter_add_(0, segment_ids, ctrl_pts[Jm] * PHI)