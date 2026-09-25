from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class BaseGA(LegacyNativeOptimizer):
    cdef public object pc
    cdef public object pm
    cdef public object selection
    cdef public object k_way
    cdef public object crossover
    cdef public object mutation
    cdef public object mutation_multipoints

    cdef void evolve(self, int epoch_c)
