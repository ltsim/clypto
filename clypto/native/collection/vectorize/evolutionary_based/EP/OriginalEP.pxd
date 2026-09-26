from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalEP(LegacyNativeOptimizer):
    cdef public object bout_size
    cdef public object n_bout_size
    cdef public object distance

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void initialize_variables(self)
    cdef void init_fields(self, NativePopulation pop)
    cdef void evolve(self, int epoch_c)
