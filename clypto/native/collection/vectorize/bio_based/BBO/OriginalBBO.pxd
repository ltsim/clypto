from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class OriginalBBO(VectorizeOptimizer):
    cdef public object p_m
    cdef public object n_elites
    cdef public object mu
    cdef public object mr

