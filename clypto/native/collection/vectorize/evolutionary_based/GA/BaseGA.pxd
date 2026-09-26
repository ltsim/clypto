from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class BaseGA(VectorizeOptimizer):
    cdef public object pc
    cdef public object pm
    cdef public object selection
    cdef public object k_way
    cdef public object crossover
    cdef public object mutation
    cdef public object mutation_multipoints

