from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalAEO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch)
