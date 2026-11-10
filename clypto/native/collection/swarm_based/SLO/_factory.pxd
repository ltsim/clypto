from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory


cdef class ModifiedSLOAgents(AgentFactory):
    """The agent factory shared by the SLO family (local best vs target at birth)."""
    cdef Agent create(self, object solution=*)
    cdef Agent generate(self, object solution=*)
