from clypto.optimizer.native.target cimport NativeTarget


cdef class Problem:
    cdef readonly object bounds
    cdef readonly str sense
    cdef readonly str name
    cdef readonly object evaluator
    cdef readonly bint vectorized
    cdef object _obj_func
    cdef object _seed
    cdef object _obj_weights
    cdef Py_ssize_t _n_objs

    cpdef object obj_func(self, object x)
    cpdef object generate_solution(self, bint encoded=*)
    cpdef object correct_solution(self, object x)
    cpdef NativeTarget get_target(self, object solution)
    cpdef object fitness(self, object objectives)
    cpdef tuple evaluate(self, object X, bint parallel=*)
