from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# cxx_args = ['-g']

# nvcc_args = [
#     '-gencode', 'arch=compute_50,code=sm_50',
#     '-gencode', 'arch=compute_52,code=sm_52',
#     '-gencode', 'arch=compute_60,code=sm_60',
#     '-gencode', 'arch=compute_61,code=sm_61',
#     '-gencode', 'arch=compute_70,code=sm_70',
#     '-gencode', 'arch=compute_70,code=sm_70',
#     '-gencode', 'arch=compute_80,code=sm_80',
# ]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
