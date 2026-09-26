from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevMVO(LegacyNativeOptimizer):
    cdef public double wep_min
    cdef public double wep_max

    cdef void evolve(self, int epoch_c)
