from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevVCS(VectorizeOptimizer):
    cdef public object lamda
    cdef public object sigma
    cdef public object n_best

