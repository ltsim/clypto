from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class OriginalLCO(LegacyNativeOptimizer):
    cdef public object r1
    cdef public object n_agents

    cdef void evolve(self, int epoch_c)
