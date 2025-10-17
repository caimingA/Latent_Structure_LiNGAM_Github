import sys
import numpy as np
A=sys.path.insert(0, "..")
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    ext_modules=cythonize('moment_estimation_c.pyx'),
    # 这句一定要有，不然只编译C代码，无法编译出pyd文件
    include_dirs=[np.get_include()]
)