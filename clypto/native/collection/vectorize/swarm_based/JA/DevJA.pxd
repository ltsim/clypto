from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevJA(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch_c)
