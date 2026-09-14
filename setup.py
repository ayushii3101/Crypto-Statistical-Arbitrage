from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "spread_engine",                        # import name in Python
        sources=[
            "src/cpp/spread_engine.cpp",        # implementation
            "src/cpp/bindings.cpp",             # pybind11 interface
        ],
        extra_compile_args=[
            "-O3",           # maximum compiler optimisation
            "-march=native", # use all CPU instructions available
                             # on this machine (SIMD, AVX etc.)
            "-std=c++17",    # C++17 standard
        ],
    ),
]

setup(
    name         = "spread_engine",
    ext_modules  = ext_modules,
    cmdclass     = {"build_ext": build_ext},
)
