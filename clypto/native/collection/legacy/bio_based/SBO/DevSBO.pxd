cimport clypto.core as cy


cdef class DevSBO(cy.Optimizer):
    cdef public double alpha
    cdef public double p_m
    cdef public double psw
