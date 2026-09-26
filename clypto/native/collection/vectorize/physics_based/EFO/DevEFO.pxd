from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevEFO(VectorizeOptimizer):
    cdef public double r_rate
    cdef public double ps_rate
    cdef public double p_field
    cdef public double n_field
    cdef public object phi

