cdef class _NogilEvaluator:
    cdef void (*_func)(const double* x, Py_ssize_t n_dims, double* out) noexcept nogil
    cdef int _n_objs
    cdef inline void row(self, const double* x, Py_ssize_t n_dims, double* out) noexcept nogil:
        self._func(x, n_dims, out)
