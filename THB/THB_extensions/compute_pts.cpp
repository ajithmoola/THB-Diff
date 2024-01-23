#include <torch/extension.h>
#include <vector>

torch::Tensor compute_pts_forward(
    torch::Tensor ctrl_pts,
    torch::Tensor Jm_array,
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
){
    const auto num_sections = num_supp_bs_cumsum.size(0);
    auto output = torch::zeros({num_sections, 3}, ctrl_pts.options());

    for (int i=0; i<num_sections; ++i){
        const int start = (i==0) ? 0 : num_supp_bs_cumsum[i - 1].item<int>();
        const int end = num_supp_bs_cumsum[i].item<int>();
        float val[3] = {0.0, 0.0, 0.0};
        for (int j=start; j<end; ++j){
            const int ctrl_idx = Jm_array[j].item<int>();
            for (int k=0; k<3; ++k){
                val[k] += tensor_prod[j].item<float>() * ctrl_pts[ctrl_idx][k].item<float>();
            }
        }
        for (int k=0; k<3; ++k){
            output[i][k] = val[k];
        }
    }

    return output;
}

torch::Tensor compute_pts_backward(
    torch::Tensor grad_output,
    torch::Tensor ctrl_pts, 
    torch::Tensor Jm_array, 
    torch::Tensor tensor_prod,
    torch::Tensor num_supp_bs_cumsum
){
    auto grad_ctrl_pts = torch::zeros_like(ctrl_pts);

    const auto num_sections = num_supp_bs_cumsum.size(0);

    for (int i=0; i<num_sections; ++i){
        const int start=(i==0) ? 0 : num_supp_bs_cumsum[i-1].item<int>();
        const int end = num_supp_bs_cumsum[i].item<int>();
        for (int j=start; j<end; ++j){
            const int ctrl_idx = Jm_array[j].item<int>();
            for (int k=0; k<3; ++k){
                grad_ctrl_pts[ctrl_idx][k] += tensor_prod[j].item<float>() * grad_output[i][k].item<float>();
            }
        }
    }

    return grad_ctrl_pts;
}

PYBIND11_MODULE(THB_eval, m) {
    m.def("cpp_forward", &compute_pts_forward, "Compute pts forward");
    m.def("cpp_backward", &compute_pts_backward, "Compute pts backward");
}