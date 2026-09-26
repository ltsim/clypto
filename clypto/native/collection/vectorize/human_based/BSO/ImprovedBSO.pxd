from clypto.optimizer.native.agent_list cimport AgentListOptimizer


cdef class ImprovedBSO(AgentListOptimizer):
    cdef public object m_clusters
    cdef public object p1
    cdef public object p2
    cdef public object p3
    cdef public object p4
    cdef public object m_solution
    cdef public object centers
    cdef public object pop_group

