cimport clypto.core as cy


cdef class OriginalES(cy.Optimizer):
    cdef public double lamda
