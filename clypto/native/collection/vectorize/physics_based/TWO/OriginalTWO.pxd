from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalTWO(LegacyNativeOptimizer):
    cdef public object muy_s
    cdef public object muy_k
    cdef public object delta_t
    cdef public object alpha
    cdef public object beta

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void init_fields(self, NativePopulation pop)
    cdef void initialization(self)
    cdef void evolve(self, int epoch_c)
