from clypto.optimizer._native.target cimport NativeTarget


cdef class NativeProblem:
    cdef readonly list bounds
    cdef readonly object lb
    cdef readonly object ub
    cdef readonly Py_ssize_t n_dims
    cdef readonly str minmax
    cdef readonly object evaluator
    cdef readonly bint vectorized
    cdef bint _all_float
    cdef object _obj_func
    cdef str _name
    cdef object _seed
    cdef object _obj_weights
    cdef Py_ssize_t _n_objs

    cpdef object obj_func(self, object x)
    cpdef object generate_solution(self, bint encoded=*)
    cpdef object correct_solution(self, object x)
    cpdef object correct_solutions(self, object X)
    cpdef NativeTarget get_target(self, object solution)
