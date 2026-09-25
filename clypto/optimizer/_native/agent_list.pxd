from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class AgentListOptimizer(LegacyNativeOptimizer):
    cdef public object objs

    cdef void initialization(self)
    cdef void after_initialization(self)
    cdef void evolve(self, int epoch)
