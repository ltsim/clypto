from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalBBO(LegacyNativeOptimizer):
    cdef public object p_m
    cdef public object n_elites
    cdef public object mu
    cdef public object mr

    cdef void evolve(self, int epoch_c)
