#

try:
    from ._cython_definitions.matrix_funcs import A, B, G, E, triad, skew_matrix as skew
except ModuleNotFoundError:
    print('Failed importing compiled Cython matrices!')
    print('Falling back to pure python mode.')
    from .numba_funcs import A, B, G, E, skew_matrix as skew
    from .misc import triad

from .numba_funcs import dcm2ep

__all__ = ['A', 'B', 'G', 'E', 'triad', 'skew', 'dcm2ep']

