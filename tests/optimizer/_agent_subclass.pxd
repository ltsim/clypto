import numpy as np
cimport numpy as cnp

from clypto.optimizer.native.agent cimport Agent


cdef class TaggedAgent(Agent):
    cdef public object tag
    cdef public double salary
    cdef public cnp.ndarray history

    cdef Agent _new_instance(self)
    cdef void _copy_state_to(self, Agent other)


cdef class NaughtyAgent(TaggedAgent):
    cdef Agent _new_instance(self)
    cdef void _copy_state_to(self, Agent other)
