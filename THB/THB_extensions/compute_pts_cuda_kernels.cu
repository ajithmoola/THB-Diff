#include <torch/extension.h>

__global__ void compute_pts_forward_kernel(
    const float* ctrl_pts,
    const long* Jm_array,
    const float* tensor_prod,
    const long* num_supp_bs_cumsum,
    float* output,
    int num_pts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pts) {
        const int start = (idx == 0) ? 0 : num_supp_bs_cumsum[idx - 1];
        const int end = num_supp_bs_cumsum[idx];
        float val[3] = {0.0, 0.0, 0.0};
        for (int j = start; j < end; ++j) {
            const int ctrl_idx = Jm_array[j];
            for (int k = 0; k < 3; ++k) {
                val[k] += tensor_prod[j] * ctrl_pts[ctrl_idx * 3 + k];
            }
        }
        for (int k = 0; k < 3; ++k) {
            output[idx * 3 + k] = val[k];
        }
    }
}

__global__ void compute_pts_backward_kernel(
    const float* grad_output,
    const long* Jm_array,
    const float* tensor_prod,
    const long* num_supp_bs_cumsum,
    float* grad_ctrl_pts,
    int num_sections
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sections) {
        const int start=(idx==0) ? 0 : num_supp_bs_cumsum[idx-1];
        const int end = num_supp_bs_cumsum[idx];
        for (int j=start; j<end; ++j) {
            const int ctrl_idx = Jm_array[j];
            for (int k = 0; k < 3; ++k) {
                atomicAdd(&grad_ctrl_pts[ctrl_idx*3+k], tensor_prod[j] * grad_output[idx*3+k]);
            }
        }
    }
}

torch::Tensor compute_pts_forward_cuda(
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
) {
    const auto num_sections = num_supp_bs_cumsum.size(0);
    auto output = torch::zeros({num_sections, 3}, ctrl_pts.options());

    const int threads = 1024;
    const int blocks = (num_sections + threads - 1) / threads;

    compute_pts_forward_kernel<<<blocks, threads>>>(
        ctrl_pts.data_ptr<float>(),
        Jm_array.data_ptr<long>(),
        tensor_prod.data_ptr<float>(),
        num_supp_bs_cumsum.data_ptr<long>(),
        output.data_ptr<float>(),
        num_sections
    );

    return output;
}

torch::Tensor compute_pts_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
) {
    auto grad_ctrl_pts = torch::zeros_like(ctrl_pts);

    const auto num_sections = num_supp_bs_cumsum.size(0);

    const int threads = 1024;
    const int blocks = (num_sections + threads - 1) / threads;

    compute_pts_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        Jm_array.data_ptr<long>(),
        tensor_prod.data_ptr<float>(),
        num_supp_bs_cumsum.data_ptr<long>(),
        grad_ctrl_pts.data_ptr<float>(),
        num_sections
    );

    return grad_ctrl_pts;
}