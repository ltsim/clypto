from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalGWO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch)
