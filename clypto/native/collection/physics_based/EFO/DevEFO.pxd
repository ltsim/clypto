cimport clypto.core as cy


cdef class DevEFO(cy.Optimizer):
    cdef public double n_field
    cdef public double p_field
    cdef public double ps_rate
    cdef public double r_rate
