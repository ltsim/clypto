from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalTWO(VectorizeOptimizer):
    cdef public object muy_s
    cdef public object muy_k
    cdef public object delta_t
    cdef public object alpha
    cdef public object beta

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void init_fields(self, NativePopulation pop)
