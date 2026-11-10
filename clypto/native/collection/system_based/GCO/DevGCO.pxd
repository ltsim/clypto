cimport clypto.core as cy


cdef class DevGCO(cy.Optimizer):
    cdef public double cr
    cdef public double wf
