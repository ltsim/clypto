from clypto.native.collection.vectorize.swarm_based.PSO._base cimport _PSOBase


cdef class OriginalPSO(_PSOBase):
    cdef public double c1
    cdef public double c2
    cdef public double w

    cdef double weight(self, int epoch)
    cdef bint clips_velocity(self)
    cdef void evolve(self, int epoch)
