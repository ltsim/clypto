from clypto.optimizer._native.agent_list cimport AgentListOptimizer


cdef class BaseGA(AgentListOptimizer):
    cdef public object pc
    cdef public object pm
    cdef public object selection
    cdef public object k_way
    cdef public object crossover
    cdef public object mutation
    cdef public object mutation_multipoints
