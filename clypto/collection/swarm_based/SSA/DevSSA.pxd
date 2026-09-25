from clypto.optimizer._native.agent_list cimport AgentListOptimizer


cdef class DevSSA(AgentListOptimizer):
    cdef public object ST
    cdef public object PD
    cdef public object SD
    cdef public object n1
    cdef public object n2

    cdef object amend_solution(self, object solution)
