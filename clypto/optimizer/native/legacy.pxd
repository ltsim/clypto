from clypto.optimizer.native.optimizer cimport NativeOptimizer


cdef class LegacyOptimizer(NativeOptimizer):
    cdef dict __dict__
    cdef public object pop
    cdef public object g_best
    cdef public object g_worst
    cdef public object problem
    cdef public object validator
