from clypto.optimizer._native.agent_list cimport AgentListOptimizer


cdef class OriginalCHIO(AgentListOptimizer):
    cdef public object brr
    cdef public object max_age
    cdef public object immunity_type_list
    cdef public object age_list
    cdef public object finished

    cdef void initialize_variables(self)
