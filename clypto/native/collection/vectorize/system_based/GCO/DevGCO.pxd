from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer


cdef class DevGCO(LegacyNativeOptimizer):
    cdef public double cr
    cdef public double wf
    cdef public object dyn_list_cell_counter
    cdef public object dyn_list_life_signal

    cdef void initialize_variables(self)
    cdef void evolve(self, int epoch)
