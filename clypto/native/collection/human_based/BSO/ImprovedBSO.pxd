cimport clypto.core as cy


cdef class ImprovedBSO(cy.Optimizer):
    cdef public int m_clusters
    cdef public double p1
    cdef public double p2
    cdef public double p3
    cdef public double p4
