# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cutlass",
    ext_modules=[
        CUDAExtension(
            "cutlass",
            [
                "cutlass.cpp",
                "cutlass_kernel.cu",
            ],
            include_dirs=[
                "/data/users/bertrand/cutlass/include",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
