from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevSBO(VectorizeOptimizer):
    cdef public object alpha
    cdef public object p_m
    cdef public object psw
    cdef public object sigma

