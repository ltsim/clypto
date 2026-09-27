cdef class NativeOptimizer:
    cdef public int epoch
    cdef public object generator
    cdef public object rng
    cdef public object tracker
    cdef public str mode
    cdef public bint sort_flag
    cdef public double EPSILON
    cdef public tuple SUPPORTED_ARRAYS
    cdef readonly str name
    cdef object _termination
    cdef tuple _params_name_ordered

    cdef void evolve(self, int epoch)

    # Engine steps, implemented by the engine (not by algorithms).
    cdef void check_problem(self, object problem, object seed)
    cdef void before_initialization(self, object starting_solutions)
    cdef void after_initialization(self)
    cdef void after_evolve(self)
