#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

py::array_t<double> assemble_Tmatrix_cpp(
    py::array_t<double> knotVec,
    py::array_t<double> newKnotVec,
    int knotVec_len,
    int newKnotVec_len,
    int p
) {
    // Get direct access to numpy array data
    auto knotVec_ptr = knotVec.unchecked<1>();
    auto newKnotVec_ptr = newKnotVec.unchecked<1>();
    
    // Initialize T1 matrix
    std::vector<std::vector<double>> T1(newKnotVec_len - 1, std::vector<double>(knotVec_len - 1, 0.0));
    
    // First iteration: build initial matrix
    for (int i = 0; i < newKnotVec_len - 1; ++i) {
        for (int j = 0; j < knotVec_len - 1; ++j) {
            if ((newKnotVec_ptr(i) >= knotVec_ptr(j)) && (newKnotVec_ptr(i) < knotVec_ptr(j + 1))) {
                T1[i][j] = 1.0;
            }
        }
    }
    
    // Iterative refinement for degrees > 0
    for (int q = 1; q <= p; ++q) {
        int T2_rows = newKnotVec_len - q - 1;
        int T2_cols = knotVec_len - q - 1;
        
        std::vector<std::vector<double>> T2(T2_rows, std::vector<double>(T2_cols, 0.0));
        
        for (int i = 0; i < T2_rows; ++i) {
            for (int j = 0; j < T2_cols; ++j) {
                double denom1 = knotVec_ptr(j + q) - knotVec_ptr(j);
                double denom2 = knotVec_ptr(j + q + 1) - knotVec_ptr(j + 1);
                
                bool zero1 = (std::abs(denom1) < 1e-14);
                bool zero2 = (std::abs(denom2) < 1e-14);
                
                if (zero1 && !zero2) {
                    // Case 1: first denominator is zero, second is not
                    T2[i][j] = (knotVec_ptr(j + q + 1) - newKnotVec_ptr(i + q)) / denom2 * T1[i][j + 1];
                }
                else if (!zero1 && zero2) {
                    // Case 2: first denominator is not zero, second is zero
                    T2[i][j] = (newKnotVec_ptr(i + q) - knotVec_ptr(j)) / denom1 * T1[i][j];
                }
                else if (!zero1 && !zero2) {
                    // Case 3: both denominators are not zero
                    double term1 = (newKnotVec_ptr(i + q) - knotVec_ptr(j)) / denom1 * T1[i][j];
                    double term2 = (knotVec_ptr(j + q + 1) - newKnotVec_ptr(i + q)) / denom2 * T1[i][j + 1];
                    T2[i][j] = term1 + term2;
                }
                // Case 4: both denominators are zero -> T2[i][j] remains 0.0
            }
        }
        
        // Update T1 with T2 for next iteration
        T1 = std::move(T2);
    }
    
    // Convert result to numpy array
    int final_rows = T1.size();
    int final_cols = T1.empty() ? 0 : T1[0].size();
    
    auto result = py::array_t<double>(
        {final_rows, final_cols},
        {sizeof(double) * final_cols, sizeof(double)}
    );
    
    auto result_ptr = result.mutable_unchecked<2>();
    
    for (int i = 0; i < final_rows; ++i) {
        for (int j = 0; j < final_cols; ++j) {
            result_ptr(i, j) = T1[i][j];
        }
    }
    
    return result;
}

PYBIND11_MODULE(tmatrix_cpp, m) {
    m.doc() = "C++ implementation of assemble_Tmatrix for performance optimization";
    
    m.def("assemble_Tmatrix_cpp", &assemble_Tmatrix_cpp, 
          "Fast C++ implementation of T-matrix assembly for B-spline refinement",
          py::arg("knotVec"), py::arg("newKnotVec"), 
          py::arg("knotVec_len"), py::arg("newKnotVec_len"), py::arg("p"));
}