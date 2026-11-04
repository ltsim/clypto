from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.optimizer cimport NativeOptimizer
from clypto.optimizer.native.population cimport Population


cdef class LegacyOptimizer(NativeOptimizer):
    cdef dict __dict__
    cdef public Population population
    cdef public Agent g_best
    cdef public Agent g_worst
    cdef public object problem
