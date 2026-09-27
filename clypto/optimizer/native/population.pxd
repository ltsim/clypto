from clypto.optimizer.native.agent cimport Agent


cdef class Population:
    cdef public list agents
    cdef public object problem
    cdef public object generator
    cdef public object idx
    cdef Py_ssize_t _size

    cpdef Population spawn(self, object agents)


cdef class ResetPopulation(Population):
    pass


cpdef Population population(object size, object range=*, type cls=*)


cdef class NativePopulation:
    cdef public object buf              # (n, width) float64, C-contiguous
    cdef double[:, ::1] view
    cdef readonly Py_ssize_t n
    cdef readonly Py_ssize_t d
    cdef readonly Py_ssize_t m
    cdef readonly Py_ssize_t width
    cdef readonly Py_ssize_t cF         # fitness column
    cdef readonly Py_ssize_t cO         # first objective column (m columns)
    cdef readonly Py_ssize_t cX         # first solution column (d columns)
    cdef dict _fields
    cdef public object weights          # objective weights of the agent snapshots

    cpdef Py_ssize_t offset(self, str name)
    cpdef Agent agent(self, Py_ssize_t i, Py_ssize_t c_obj=*)
    cpdef NativePopulation take(self, object rows)
    cpdef NativePopulation concat(self, NativePopulation other)
    cpdef NativePopulation empty_like(self)


cdef inline double np_clip(double x, double lo, double hi) noexcept nogil:
    # np.clip for float64: NaN in x propagates; otherwise min(max(x, lo), hi).
    if x != x:
        return x
    x = x if x > lo else lo
    if x != x:
        return x
    return x if x < hi else hi


cdef void select_better(
    NativePopulation dst, Py_ssize_t dst_x, Py_ssize_t dst_o, Py_ssize_t dst_f,
    NativePopulation src, Py_ssize_t src_x, Py_ssize_t src_o, Py_ssize_t src_f,
    Py_ssize_t start, Py_ssize_t stop, bint maximize,
) noexcept nogil
