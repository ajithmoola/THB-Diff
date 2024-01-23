#include <pybind11/pybind11.h>
#include <array>

namespace py = pybind11;

PYBIND11_MODULE(funcs, m) {
    m.doc() = "THB-spline domain octree datastructure";
    m.def("add", &add, "example function");
}