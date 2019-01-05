

cdef void A_cy(double[:,:] arr, double[:,:] result) nogil
cdef A(double[:,:] arr)

cdef void B_cy(double[:,:] p, double[:,:] u, double[:,:] result) nogil
cdef B(double[:,:] p, double[:,:] u)

cdef skew_matrix(double[:,:] v)
cdef orthogonal_vector(double[:,:] v)
cdef triad(double[:,:] v1, double[:,:] v2=?)

cdef sparse_assembler(list blocks, int[:] b_rows, int[:] b_cols,
                       list e_data, list e_rows, list e_cols)

