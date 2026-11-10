from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.problem cimport Problem


cdef class AgentFactory:
    """The per-algorithm lifecycle of an agent, held by Population (see factory.pyx)."""
    cdef Problem problem
    cdef object generator

    cdef void bind(self, Problem problem, object generator)
    cdef Agent create(self, object solution=*)
    cdef Agent generate(self, object solution=*)
