from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevFBIO(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch_c)
