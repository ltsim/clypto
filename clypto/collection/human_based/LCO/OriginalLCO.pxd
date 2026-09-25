from clypto.optimizer._native.agent_list cimport AgentListOptimizer


cdef class OriginalLCO(AgentListOptimizer):
    cdef public object r1
    cdef public object n_agents
