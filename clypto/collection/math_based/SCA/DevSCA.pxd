from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class DevSCA(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch)
