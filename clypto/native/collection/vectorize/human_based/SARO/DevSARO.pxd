from clypto.optimizer.native.agent_list cimport AgentListOptimizer


cdef class DevSARO(AgentListOptimizer):
    cdef public object se
    cdef public object mu
    cdef public object dyn_USN

    cdef object amend_solution(self, object solution)
