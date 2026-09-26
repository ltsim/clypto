cdef class NativeTarget:
    cdef readonly object objectives
    cdef readonly object weights
    cdef readonly double fitness

    cpdef NativeTarget copy(self)
