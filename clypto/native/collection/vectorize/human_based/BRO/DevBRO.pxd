from clypto.optimizer.native.agent_list cimport AgentListOptimizer


cdef class DevBRO(AgentListOptimizer):
    cdef public object threshold
    cdef public object dyn_delta

    cdef public object lb_updated
    cdef public object ub_updated

    cdef void initialize_variables(self)
