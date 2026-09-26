from clypto.optimizer.native.agent cimport LegacyNativeAgent
from clypto.optimizer.native.optimizer cimport NativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.problem cimport Problem


cdef class VectorizeOptimizer(NativeOptimizer):
    cdef public LegacyNativeAgent g_best
    cdef public LegacyNativeAgent g_worst
    cdef public NativePopulation pop
    cdef public Problem problem
    cdef Py_ssize_t _g_best_row     # row aliased by g_best (-1: g_best is a copy)
    cdef object _starting

    # Storage hooks: override with the same cdef signature.
    cdef list layout(self, Py_ssize_t d, Py_ssize_t m)
    cdef void init_fields(self, NativePopulation pop)
    cdef object _amend_solution(self, object solution)

    cpdef object _correct_solution(self, object solution)
    cdef NativePopulation generate_population(self, Py_ssize_t k)
    cdef NativePopulation new_population(self, object X)
    cdef void evaluate(self, NativePopulation pop, Py_ssize_t start, Py_ssize_t stop)
    cdef object sorted_order(self, NativePopulation pop)
    cdef object g_best_x(self)
    cdef LegacyNativeAgent current_g_best(self)
