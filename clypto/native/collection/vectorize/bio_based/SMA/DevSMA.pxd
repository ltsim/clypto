from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevSMA(LegacyNativeOptimizer):
    cdef public object p_t
    cdef public object weights

    cdef void initialize_variables(self)
    cdef void evolve(self, int epoch_c)
