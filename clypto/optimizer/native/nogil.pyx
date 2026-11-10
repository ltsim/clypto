#!/usr/bin/env python
# Created for clypto's nogil parallel evaluation path.
# --------------------------------------------------%
"""Private nogil objective contract for ``mode="parallel"``.

Real parallelism requires an objective that can run without the GIL. A pure
Python ``obj_func`` cannot, so the parallel path stays sequential unless the
problem carries an evaluator built from a compiled nogil function. The team is
fixed once by the engine's ``solve`` (``omp_set_num_threads``); a region never
reconfigures it.

Compiled users subclass :class:`_NogilEvaluator` in a ``.pyx`` module, cimport
it from ``clypto.optimizer.native.nogil``, and assign their function to ``_func``:

    from clypto.optimizer.native.nogil cimport _NogilEvaluator

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

and pass ``SphereEvaluator()`` as ``Problem(..., evaluator=...)``. Multi-objective
evaluators set ``_n_objs`` and write one value per objective into ``out``; the
``Problem`` validates both counts when the evaluation runs.
"""
cdef class _NogilEvaluator:
    def __cinit__(self):
        self._func = NULL
        self._n_objs = 1

    @property
    def n_objs(self):
        """Objectives written per row: ``_func`` fills ``out[0] ... out[n_objs-1]``."""
        return self._n_objs
