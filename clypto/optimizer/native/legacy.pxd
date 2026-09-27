from clypto.optimizer.native.optimizer cimport NativeOptimizer
from clypto.optimizer.native.population cimport Population


cdef class LegacyOptimizer(NativeOptimizer):
    cdef dict __dict__
    cdef Population _population
    cdef public object g_best
    cdef public object g_worst
    cdef public object problem
