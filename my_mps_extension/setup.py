from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Get the directory of this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="my_mps_extension",
    packages=[],  # 明确指定不包含任何 Python 包
    ext_modules=[
        CppExtension(
            name="my_mps_extension._C",
            sources=[
                os.path.join(current_dir, "causal_conv1d_kernel.mm")
            ],
            extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
            extra_compile_args={
                'cxx': ['-std=c++20', '-O3']
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)


