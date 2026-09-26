from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalES(VectorizeOptimizer):
    cdef public object lamda
    cdef public object n_child
    cdef public object distance

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void init_fields(self, NativePopulation pop)
