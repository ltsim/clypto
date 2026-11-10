cdef class Problem:
    cdef object __bounds
    cdef str __sense
    cdef str __name
    cdef object __evaluator
    cdef bint __vectorized
    cdef bint __decode
    cdef long long __n_evals
    cdef object __obj_func
    cdef object __seed
    cdef object __obj_weights
    cdef Py_ssize_t __n_objs

    cpdef object obj_func(self, object x)
    cpdef object generate_solution(self, bint encoded=*)
    cpdef object correct_solution(self, object x)
    cpdef object evaluate_solution(self, object solution)
    cpdef object fitness(self, object objectives)
    cpdef tuple evaluate(self, object X, bint parallel=*, int threads=*)
