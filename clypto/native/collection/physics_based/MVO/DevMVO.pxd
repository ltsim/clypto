cimport clypto.core as cy


cdef class DevMVO(cy.Optimizer):
    cdef public double wep_max
    cdef public double wep_min
