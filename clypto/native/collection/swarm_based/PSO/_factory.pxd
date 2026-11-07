from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory


cdef class PSOAgents(AgentFactory):
    """The particle factory shared by the PSO family (velocity + pbest = self)."""
    cdef public object owner

    cdef Agent create(self, object solution=*)
    cdef Agent generate(self, object solution=*)
