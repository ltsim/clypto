from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalEP(VectorizeOptimizer):
    cdef public object bout_size
    cdef public object n_bout_size
    cdef public object distance

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void init_fields(self, NativePopulation pop)
