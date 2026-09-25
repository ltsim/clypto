"""Structure-of-arrays population for the native collection.

Every agent is one row of a single C-contiguous ``float64`` buffer::

    [ F | O (m) | X (d) | algorithm fields ... ]

Algorithm fields are declared as ``(name, width)`` pairs (e.g. PSO's velocity,
personal best and its objectives/fitness) and live in the same row, so copying
an agent is copying a row and sorting is one fancy-index.
"""
import numpy as np

from cython.parallel cimport prange

# Rows x columns below which OpenMP threads cost more than they save.
cdef Py_ssize_t PARALLEL_MIN_WORK = 20000


cdef class NativePopulation:
    def __init__(self, Py_ssize_t n, Py_ssize_t d, Py_ssize_t m, fields=(), weights=None):
        cdef Py_ssize_t col = 1 + m + d
        self.n, self.d, self.m = n, d, m
        self.cF, self.cO, self.cX = 0, 1, 1 + m
        self._fields = {}
        for name, size in fields:
            self._fields[name] = (col, size)
            col += size
        self.width = col
        self.buf = np.full((n, col), np.nan)
        self.view = self.buf
        self.weights = weights

    cpdef Py_ssize_t offset(self, str name):
        return self._fields[name][0]

    def field(self, str name):
        """``(n, width)`` view of a declared field."""
        off, size = self._fields[name]
        return self.buf[:, off:off + size]

    @property
    def X(self):
        return self.buf[:, self.cX:self.cX + self.d]

    @property
    def O(self):
        return self.buf[:, self.cO:self.cO + self.m]

    @property
    def F(self):
        return self.buf[:, self.cF]

    cpdef NativeTarget target_at(self, Py_ssize_t i, Py_ssize_t c_obj):
        return NativeTarget(self.buf[i, c_obj:c_obj + self.m].copy(), self.weights)

    cpdef LegacyNativeAgent agent(self, Py_ssize_t i):
        """Snapshot of row ``i`` as a standalone agent."""
        return LegacyNativeAgent(
            self.buf[i, self.cX:self.cX + self.d].copy(), self.target_at(i, self.cO)
        )

    cpdef NativePopulation take(self, object rows):
        """New population holding copies of ``rows`` (in that order)."""
        cdef NativePopulation out = self.empty_like()
        out.buf = np.ascontiguousarray(self.buf[rows])
        out.view = out.buf
        out.n = out.buf.shape[0]
        return out

    cpdef NativePopulation concat(self, NativePopulation other):
        """Rows of ``self`` followed by the rows of ``other`` (same layout)."""
        cdef NativePopulation out = self.empty_like()
        out.buf = np.concatenate([self.buf, other.buf])
        out.view = out.buf
        out.n = out.buf.shape[0]
        return out

    cpdef NativePopulation empty_like(self):
        cdef NativePopulation out = NativePopulation.__new__(NativePopulation)
        out.n, out.d, out.m, out.width = self.n, self.d, self.m, self.width
        out.cF, out.cO, out.cX = self.cF, self.cO, self.cX
        out._fields = self._fields
        out.weights = self.weights
        out.buf = np.full((self.n, self.width), np.nan)
        out.view = out.buf
        return out

    def __len__(self):
        return self.n

    def __getitem__(self, Py_ssize_t i):
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.agent(i)

    def __iter__(self):
        return (self.agent(i) for i in range(self.n))


cdef void select_better(
    NativePopulation dst, Py_ssize_t dst_x, Py_ssize_t dst_o, Py_ssize_t dst_f,
    NativePopulation src, Py_ssize_t src_x, Py_ssize_t src_o, Py_ssize_t src_f,
    Py_ssize_t start, Py_ssize_t stop, bint maximize,
) noexcept nogil:
    """Copy src rows into dst where src is better (``cy.compare_target`` semantics).

    min: ``new < old``; max: ``not (new < old)``. Rows are independent, so the
    result does not depend on the number of threads.
    """
    cdef double[:, ::1] a = dst.view
    cdef double[:, ::1] b = src.view
    cdef Py_ssize_t i, j, d = dst.d, m = dst.m
    cdef bint better
    for i in prange(start, stop, schedule="static",
                    use_threads_if=(stop - start) * (d + m) > PARALLEL_MIN_WORK):
        better = b[i, src_f] < a[i, dst_f]
        if maximize:
            better = not better
        if better:
            for j in range(d):
                a[i, dst_x + j] = b[i, src_x + j]
            for j in range(m):
                a[i, dst_o + j] = b[i, src_o + j]
            a[i, dst_f] = b[i, src_f]
