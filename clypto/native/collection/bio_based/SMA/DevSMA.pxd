cimport clypto.core as cy


cdef class DevSMA(cy.Optimizer):
    cdef public double p_t
