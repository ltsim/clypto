from clypto.optimizer.native.population cimport NativePopulation
from clypto.native.collection.vectorize.swarm_based.PSO._base cimport _PSOBase


cdef class P_PSO(_PSOBase):
    cdef public object dyn_delta_list

    cdef void initialize_variables(self)
    cdef void evolve(self, int epoch)
