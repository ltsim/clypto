from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalFOA(LegacyNativeOptimizer):
    cdef void initialization(self)
    cdef void evolve(self, int epoch_c)
