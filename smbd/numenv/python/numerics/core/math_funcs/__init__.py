#

try:
    from .matrix_funcs import A, B, G, E, triad, skew_matrix as skew
except ModuleNotFoundError:
    print('Failed importing compiled Cython matrices!')
    print('Falling back to pure python mode.')
    from .misc import A, B, G, E, triad, skew_matrix as skew


__all__ = ['A', 'B', 'G', 'E', 'triad', 'skew']

