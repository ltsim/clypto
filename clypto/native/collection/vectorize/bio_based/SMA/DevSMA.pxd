from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevSMA(VectorizeOptimizer):
    cdef public object p_t
    cdef public object weights

