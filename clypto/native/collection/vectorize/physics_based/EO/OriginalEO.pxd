from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class OriginalEO(VectorizeOptimizer):
    cdef public object V
    cdef public object a1
    cdef public object a2
    cdef public object GP

