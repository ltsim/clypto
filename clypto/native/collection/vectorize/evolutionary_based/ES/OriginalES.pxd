from clypto.optimizer._native.agent_list cimport AgentListOptimizer


cdef class OriginalES(AgentListOptimizer):
    cdef public object lamda
    cdef public object n_child
    cdef public object distance

    cdef void initialize_variables(self)
