from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevMVO(VectorizeOptimizer):
    cdef public double wep_min
    cdef public double wep_max

