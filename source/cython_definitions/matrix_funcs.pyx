# distutils: extra_compile_args=-openmp
# distutils: extra_link_args=-openmp

cimport cython
cimport numpy as cnp
import numpy as np
from cpython cimport array
from numpy.linalg import multi_dot, norm
from cython.parallel import prange, parallel
from libc.math cimport fabs, exp

ctypedef cnp.float64_t DTYPE_t


@cython.wraparound (False)
@cython.boundscheck(False)
cdef void A_cy(double[:,:] arr, double[:,:] result) nogil:

    cdef double e0 = arr[0,0]
    cdef double e1 = arr[1,0]
    cdef double e2 = arr[2,0]
    cdef double e3 = arr[3,0]
        
    result[0,0] = (e0**2+e1**2-e2**2-e3**2)
    result[0,1] = 2*((e1*e2)-(e0*e3))              
    result[0,2] = 2*((e1*e3)+(e0*e2))
    
    result[1,0] = 2*((e1*e2)+(e0*e3))
    result[1,1] = e0**2-e1**2+e2**2-e3**2
    result[1,2] = 2*((e2*e3)-(e0*e1))
    
    result[2,0] = 2*((e1*e3)-(e0*e2))
    result[2,1] = 2*((e2*e3)+(e0*e1))
    result[2,2] = e0**2-e1**2-e2**2+e3**2
    
cpdef A(double[:,:] arr):
    m = np.zeros((3,3),dtype=np.float64)
    A_cy(arr, m)
    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef ep2dcm(double[:,:] arr):
    
    cdef double e0 = arr[0,0]
    cdef double e1 = arr[1,0]
    cdef double e2 = arr[2,0]
    cdef double e3 = arr[3,0]
    
    m = np.empty((3,3),dtype=np.float64)
    cdef double[:,:] result = m.view()
    
    result[0,0] = (e0**2+e1**2-e2**2-e3**2)
    result[0,1] = 2*((e1*e2)-(e0*e3))              
    result[0,2] = 2*((e1*e3)+(e0*e2))
    
    result[1,0] = 2*((e1*e2)+(e0*e3))
    result[1,1] = e0**2-e1**2+e2**2-e3**2
    result[1,2] = 2*((e2*e3)-(e0*e1))
    
    result[2,0] = 2*((e1*e3)-(e0*e2))
    result[2,1] = 2*((e2*e3)+(e0*e1))
    result[2,2] = e0**2-e1**2-e2**2+e3**2
    
    return m


@cython.wraparound (False)
@cython.boundscheck(False)
cpdef B(double[:,:] p, double[:,:] u):
    
    cdef double e0 = p[0,0]
    cdef double e1 = p[1,0]
    cdef double e2 = p[2,0]
    cdef double e3 = p[3,0]
    
    cdef double ux = u[0,0]
    cdef double uy = u[1,0]
    cdef double uz = u[2,0]
    
    m = np.empty((3,4),dtype=np.float64)
    cdef double[:,:] result = m.view()
    
    result[0,0] = 2*e0*ux + 2*e2*uz - 2*e3*uy
    result[0,1] = 2*e1*ux + 2*e2*uy + 2*e3*uz
    result[0,2] = 2*e0*uz + 2*e1*uy - 2*e2*ux
    result[0,3] = -2*e0*uy + 2*e1*uz - 2*e3*ux
    
    result[1,0] = 2*e0*uy - 2*e1*uz + 2*e3*ux
    result[1,1] = -2*e0*uz - 2*e1*uy + 2*e2*ux
    result[1,2] = 2*e1*ux + 2*e2*uy + 2*e3*uz
    result[1,3] = 2*e0*ux + 2*e2*uz - 2*e3*uy
    
    result[2,0] = 2*e0*uz + 2*e1*uy - 2*e2*ux
    result[2,1] = 2*e0*uy - 2*e1*uz + 2*e3*ux
    result[2,2] = -2*e0*ux - 2*e2*uz + 2*e3*uy
    result[2,3] = 2*e1*ux + 2*e2*uy + 2*e3*uz

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef E(double[:,:] p):
    
    cdef double e0 = p[0,0]
    cdef double e1 = p[1,0]
    cdef double e2 = p[2,0]
    cdef double e3 = p[3,0]
        
    m = np.empty((3,4),dtype=np.float64)
    cdef double[:,:] result = m.view()
    
    result[0,0] = -e1
    result[0,1] =  e0
    result[0,2] = -e3
    result[0,3] =  e2
    
    result[1,0] = -e2
    result[1,1] =  e3
    result[1,2] =  e0
    result[1,3] = -e1
    
    result[2,0] = -e3
    result[2,1] = -e2
    result[2,2] =  e1
    result[2,3] =  e0

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef G(double[:,:] p):
    
    cdef double e0 = p[0,0]
    cdef double e1 = p[1,0]
    cdef double e2 = p[2,0]
    cdef double e3 = p[3,0]
        
    m = np.empty((3,4),dtype=np.float64)
    cdef double[:,:] result = m.view()
    
    result[0,0] = -e1
    result[0,1] =  e0
    result[0,2] =  e3
    result[0,3] = -e2
    
    result[1,0] = -e2
    result[1,1] = -e3
    result[1,2] =  e0
    result[1,3] =  e1
    
    result[2,0] = -e3
    result[2,1] =  e2
    result[2,2] = -e1
    result[2,3] =  e0

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef skew_matrix(double[:,:] v):
    cdef double x = v[0,0]
    cdef double y = v[1,0]
    cdef double z = v[2,0]
    
    m = np.zeros((3,3),dtype=np.float64)
    cdef double[:,:] result = m.view()
    
    result[0,1] = -z
    result[0,2] = y
    result[1,0] = z
    result[1,2] = -x
    result[2,0] = -y
    result[2,1] = x

    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef orthogonal_vector(double[:,:] v):
    
    dummy = np.ones((3,1),dtype=np.float64)
    cdef int i = np.argmax(np.abs(v))
    dummy[i] = 0
    m = multi_dot([skew_matrix(v),dummy])
    return m

@cython.wraparound (False)
@cython.boundscheck(False)
cpdef triad(double[:,:] v1, double[:,:] v2=None):
    cdef double[:,:] k 
    cdef double[:,:] i 
    cdef double[:,:] j 
    
    k = v1/norm(v1).view()
    
    if v2 is not None:
        i = v2/norm(v2).view()
    else:
        i = orthogonal_vector(k).view()
        i = i/norm(i).view()

    j = multi_dot([skew_matrix(k),i]).view()
    j = j/norm(j).view()
    
    m = np.empty((3,3),dtype=np.float64)
    cdef double[:,:] result = m.view()
    
    result[0,0] = i[0,0]
    result[0,1] = j[0,0]              
    result[0,2] = k[0,0]
    
    result[1,0] = i[1,0]
    result[1,1] = j[1,0]
    result[1,2] = k[1,0]
    
    result[2,0] = i[2,0]
    result[2,1] = j[2,0]
    result[2,2] = k[2,0]
    
    return m

    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef sparse_assembler(list blocks, int[:] b_rows, int[:] b_cols,
                       list e_data, list e_rows, list e_cols): 
    cdef:
        int row_counter = 0 
        int col_counter = 0
        int prev_rows_size = 0
        int prev_cols_size = 0
        int flat_count = 0
        int nnz = len(b_rows)
        int v, i, j, vi, vj, m, n
        double value
        double[:,:] arr
    
    
    for v in range(nnz):
        vi = b_rows[v]
        vj = b_cols[v]
        
        if vi != row_counter:
            row_counter +=1
            prev_rows_size += m
            prev_cols_size  = 0
        
        arr = blocks[v].view()
        m = arr.shape[0]
        n = arr.shape[1]
        
        if n==3:
            prev_cols_size = 7*(vj//2)
        elif n==4:
            prev_cols_size = 7*(vj//2)+3
        
        for i in range(m):
            for j in range(n):
                value = arr[i,j]
                if fabs(value)>exp(-2):
                    e_rows.append(prev_rows_size+i)
                    e_cols.append(prev_cols_size+j)
                    e_data.append(value)
        
        