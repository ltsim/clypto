cimport clypto.core as cy


cdef class DevSSA(cy.Optimizer):
    cdef public double PD
    cdef public double SD
    cdef public double ST
