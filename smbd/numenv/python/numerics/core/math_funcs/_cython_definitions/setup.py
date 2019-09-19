from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='matrix_funcs',
      ext_modules=cythonize("matrix_funcs.pyx", compiler_directives={'language_level' : "3"}),
      include_dirs = [np.get_include()])