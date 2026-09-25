from clypto.optimizer._native.target cimport NativeTarget

cdef class _LegacyAgent:
    cdef public object solution
    cdef public object target

    cpdef object copy(self)
    cpdef void update_agent(self, solution, target)
    cpdef bint sync_if_duplicate(self, _LegacyAgent other)
    cdef int _compare_fitness(self, _LegacyAgent other, str minmax)
    cpdef object get_better_solution(self, _LegacyAgent other, str minmax=*)
    cpdef bint is_better_than(self, _LegacyAgent other, str minmax=*)


cdef class LegacyNativeAgent:
    cdef public object solution
    cdef public NativeTarget target

    cpdef LegacyNativeAgent copy(self)
    cpdef bint sync_if_duplicate(self, LegacyNativeAgent other)
    cdef int _compare_fitness(self, LegacyNativeAgent other, str minmax)
    cpdef LegacyNativeAgent get_better_solution(self, LegacyNativeAgent other, str minmax=*)
    cpdef bint is_better_than(self, LegacyNativeAgent other, str minmax=*)
