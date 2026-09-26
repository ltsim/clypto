from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class _PSOBase(LegacyNativeOptimizer):
    cdef public object v_max
    cdef public object v_min
    cdef Py_ssize_t cV, cP, cPO, cPF

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void initialize_variables(self)
    cdef void init_fields(self, NativePopulation pop)
    cdef list chunks(self, Py_ssize_t n)
    cdef void accept(self, NativePopulation cand, Py_ssize_t start, Py_ssize_t stop)
    cdef object amend_random(self, object pos, object U)
