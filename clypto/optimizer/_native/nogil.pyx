#!/usr/bin/env python
# Created for clypto's nogil parallel evaluation path.
# --------------------------------------------------%
"""Private nogil objective contract for ``mode="parallel"``.

Real parallelism requires an objective that can run without the GIL. A pure
Python ``obj_func`` cannot, so the parallel path stays sequential unless the
problem carries an evaluator built from a compiled nogil function.

Compiled users subclass :class:`_NogilEvaluator` in a ``.pyx`` module, cimport
it from ``clypto.optimizer._native.nogil``, and assign their function to ``_func``:

    from clypto.optimizer._native.nogil cimport _NogilEvaluator

    cdef void sphere(const double* x, Py_ssize_t n, double* out) noexcept nogil:
        cdef Py_ssize_t i
        cdef double s = 0.0
        for i in range(n):
            s += x[i] * x[i]
        out[0] = s

    cdef class SphereEvaluator(_NogilEvaluator):
        def __cinit__(self):
            self._func = sphere
            self._n_objs = 1

and pass ``SphereEvaluator()`` as ``Problem(..., evaluator=...)``.
"""
cdef class _NogilEvaluator:
    def __cinit__(self):
        self._func = NULL
        self._n_objs = 1
