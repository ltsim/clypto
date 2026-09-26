from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalGWO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch)
