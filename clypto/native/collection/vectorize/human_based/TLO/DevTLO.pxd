from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevTLO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch_c)
