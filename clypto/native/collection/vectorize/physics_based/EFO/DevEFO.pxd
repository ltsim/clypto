from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevEFO(LegacyNativeOptimizer):
    cdef public double r_rate
    cdef public double ps_rate
    cdef public double p_field
    cdef public double n_field
    cdef public object phi

    cdef void evolve(self, int epoch_c)
