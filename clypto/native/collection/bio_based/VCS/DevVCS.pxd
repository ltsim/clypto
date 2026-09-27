cimport clypto.core as cy


cdef class DevVCS(cy.Optimizer):
    cdef public double lamda
    cdef public double sigma
