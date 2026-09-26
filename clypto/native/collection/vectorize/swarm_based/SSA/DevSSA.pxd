from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer


cdef class DevSSA(LegacyNativeOptimizer):
    cdef public object ST
    cdef public object PD
    cdef public object SD
    cdef public object n1
    cdef public object n2

    cdef object amend_solution(self, object solution)
    cdef void evolve(self, int epoch_c)
