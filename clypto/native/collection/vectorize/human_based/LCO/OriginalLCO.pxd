from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class OriginalLCO(VectorizeOptimizer):
    cdef public object r1
    cdef public object n_agents

