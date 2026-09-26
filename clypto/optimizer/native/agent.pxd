from clypto.optimizer.native.target cimport NativeTarget

cdef class LegacyAgent:
    cdef public object solution
    cdef public object target


cdef class LegacyNativeAgent:
    cdef public object solution
    cdef public NativeTarget target


cpdef object duplicate_agent(object agent)
cpdef bint sync_if_duplicate(object that, object other)
cpdef int compare_fitness(object that, object other, str sense=*)
cpdef object get_better_solution(object that, object other, str sense=*)
cpdef bint is_better_than(object that, object other, str sense=*)
