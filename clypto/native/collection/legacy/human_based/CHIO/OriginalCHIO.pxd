cimport clypto.core as cy


cdef class OriginalCHIO(cy.Optimizer):
    cdef public double brr
    cdef public int max_age
