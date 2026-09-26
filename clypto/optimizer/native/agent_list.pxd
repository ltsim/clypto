from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class AgentListOptimizer(VectorizeOptimizer):
    cdef public object objs
