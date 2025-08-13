from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="causal_conv1d_mps",
    packages=[],
    ext_modules=[
        CppExtension(
            name="causal_conv1d_mps._C",
            sources=["causal_conv1d_kernel.mm"],
            extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
            extra_compile_args={"cxx": ["-std=c++20", "-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
