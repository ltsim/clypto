cdef class _LegacyAgent:
    cdef public object solution
    cdef public object target

    cpdef object copy(self)
    cpdef void update_agent(self, solution, target)
    cpdef bint sync_if_duplicate(self, _LegacyAgent other)
    cdef int _compare_fitness(self, _LegacyAgent other, str minmax)
    cpdef object get_better_solution(self, _LegacyAgent other, str minmax=*)
    cpdef bint is_better_than(self, _LegacyAgent other, str minmax=*)
