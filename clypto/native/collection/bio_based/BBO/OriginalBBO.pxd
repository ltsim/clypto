cimport clypto.core as cy


cdef class OriginalBBO(cy.Optimizer):
    cdef public int n_elites
    cdef public double p_m
