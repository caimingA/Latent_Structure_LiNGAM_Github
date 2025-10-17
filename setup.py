# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="moment_estimation_c",
        sources=["moment_estimation_c.pyx"],
        extra_compile_args=["-O3"],       # 可选：加速选项
        include_dirs=[],                  # 如果有 C 头文件依赖可以加路径
    )
]

setup(
    name="moment_estimation_c",
    ext_modules=cythonize(extensions, language_level=3),
)