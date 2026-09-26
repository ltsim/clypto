cdef class NativeOptimizer:
    cdef public int epoch
    cdef public int pop_size
    cdef public object generator
    cdef public object rng
    cdef public object tracker
    cdef public object mode
    cdef public bint sort_flag
    cdef public double EPSILON
    cdef public tuple AVAILABLE_MODES
    cdef public tuple SUPPORTED_ARRAYS
    cdef readonly str name
    cdef object _termination
    cdef tuple _params_name_ordered
    cdef object _last_gbest_fit
    cdef long long _nfe_counter
    cdef int _repeated_times

    cdef void _update_repeated_times(self)
