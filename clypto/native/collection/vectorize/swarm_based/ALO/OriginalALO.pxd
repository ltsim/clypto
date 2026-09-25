from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalALO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch_c)
