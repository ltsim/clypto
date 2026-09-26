from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalAEO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch_c)
