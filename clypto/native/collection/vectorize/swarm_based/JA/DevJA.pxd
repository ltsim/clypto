from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class DevJA(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch_c)
