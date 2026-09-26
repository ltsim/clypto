from clypto.optimizer.native.vectorize cimport VectorizeOptimizer


cdef class DevSSA(VectorizeOptimizer):
    cdef public object ST
    cdef public object PD
    cdef public object SD
    cdef public object n1
    cdef public object n2

    cdef object _amend_solution(self, object solution)
