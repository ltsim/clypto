cimport clypto.core as cy


cdef class P_PSO(cy.Optimizer):
    cdef public object v_max
    cdef public object v_min
