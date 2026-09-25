from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class DevHS(LegacyNativeOptimizer):
    cdef public double c_r
    cdef public double pa_r
    cdef public object fw
    cdef public object fw_damp
    cdef public object dyn_fw

    cdef void initialize_variables(self)
    cdef void evolve(self, int epoch)
