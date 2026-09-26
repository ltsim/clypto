from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevVCS(LegacyNativeOptimizer):
    cdef public object lamda
    cdef public object sigma
    cdef public object n_best

    cdef void evolve(self, int epoch_c)
