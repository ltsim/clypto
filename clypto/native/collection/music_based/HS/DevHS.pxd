cimport clypto.core as cy


cdef class DevHS(cy.Optimizer):
    cdef public double c_r
    cdef public double pa_r
