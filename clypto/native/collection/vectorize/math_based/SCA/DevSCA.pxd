from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevSCA(LegacyNativeOptimizer):
    cdef void evolve(self, int epoch)
