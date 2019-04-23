from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(name='matrix_funcs',
      ext_modules=cythonize("matrix_funcs.pyx"),
      include_dirs = [np.get_include()],)