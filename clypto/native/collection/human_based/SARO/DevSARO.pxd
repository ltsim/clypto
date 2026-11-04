cimport clypto.core as cy


cdef class DevSARO(cy.Optimizer):
    cdef public int mu
    cdef public double se
