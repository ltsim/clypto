cdef class _LegacyOptimizer:
    cdef dict __dict__
    cdef public int epoch
    cdef public int pop_size
    cdef public object g_best
    cdef public object tracker
    cdef public object pop
    cdef public object g_worst
    cdef public object problem
    cdef public object generator
    cdef public object validator
    cdef public object EPSILON
    cdef public object AVAILABLE_MODES
    cdef public object SUPPORTED_ARRAYS
    cdef public object mode
    cdef public object rng
    cdef public object parameters
    cdef object _termination
    cdef object _name
    cdef object _params_name_ordered
    cdef object _last_gbest_fit
    cdef long long _nfe_counter
    cdef int _repeated_times
