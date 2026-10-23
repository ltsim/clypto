cimport numpy as cnp
cimport clypto.core as cy


cdef class PSOAgent(cy.Agent):
    cdef public cnp.ndarray velocity
    cdef public cnp.ndarray pbest_solution
    cdef public double pbest_fitness


cdef class PSOPopulation(cy.Population):
    cdef public cnp.ndarray v_max
    cdef public cnp.ndarray v_min
