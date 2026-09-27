cimport clypto.core as cy


cdef class OriginalCRO(cy.Optimizer):
    cdef public double Fa
    cdef public double Fb
    cdef public double Fd
    cdef public double GCR
    cdef public double Pd
    cdef public double gamma_max
    cdef public double gamma_min
    cdef public int n_trials
    cdef public double po
