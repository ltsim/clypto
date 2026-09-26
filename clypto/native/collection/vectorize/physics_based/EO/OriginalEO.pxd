from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalEO(LegacyNativeOptimizer):
    cdef public object V
    cdef public object a1
    cdef public object a2
    cdef public object GP

    cdef void evolve(self, int epoch_c)
