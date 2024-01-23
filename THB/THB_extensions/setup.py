from setuptools import setup, Extension
from torch.utils import cpp_extension

module_name = "THB_eval"

try:
    setup(
        name=module_name,
        ext_modules=[
            cpp_extension.CUDAExtension(
                name=module_name,
                sources=['compute_pts_cuda.cpp', 'compute_pts_cuda_kernels.cu'],
                include_dirs=cpp_extension.include_paths(),
                extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
            ),
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )
except:
    print('CUDA device not available, installing CPU version of THBDiff')
    setup(
        name=module_name,
        ext_modules=[
            cpp_extension.CppExtension(
                name=module_name,
                sources=['compute_pts.cpp'],
                include_dirs=cpp_extension.include_paths(),
                extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}
            ),
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension},
    )