from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import numpy as np
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "THBDiff.tmatrix_cpp",
        ["THBDiff/tmatrix_cpp.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        cxx_std=14,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
