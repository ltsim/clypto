from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevSBO(LegacyNativeOptimizer):
    cdef public object alpha
    cdef public object p_m
    cdef public object psw
    cdef public object sigma

    cdef void evolve(self, int epoch_c)
