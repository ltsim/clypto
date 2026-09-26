from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevHS(VectorizeOptimizer):
    cdef public double c_r
    cdef public double pa_r
    cdef public object fw
    cdef public object fw_damp
    cdef public object dyn_fw

