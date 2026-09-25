from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalES(LegacyNativeOptimizer):
    cdef public object lamda
    cdef public object n_child
    cdef public object distance

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void initialize_variables(self)
    cdef void init_fields(self, NativePopulation pop)
    cdef void evolve(self, int epoch_c)
