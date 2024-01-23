#include <torch/extension.h>
#include <vector>

torch::Tensor compute_pts_forward_cuda(
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
);

torch::Tensor compute_pts_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
);

torch::Tensor compute_pts_forward(
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
) {
    return compute_pts_forward_cuda(ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum);
}

torch::Tensor compute_pts_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
) {
    return compute_pts_backward_cuda(grad_output, ctrl_pts, Jm_array, tensor_prod, num_supp_bs_cumsum);
}

PYBIND11_MODULE(THB_eval, m) {
    m.def("forward", &compute_pts_forward, "Compute pts forward");
    m.def("backward", &compute_pts_backward, "Compute pts backward");
}
