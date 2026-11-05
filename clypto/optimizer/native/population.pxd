import cython

cimport numpy as cnp

from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.problem cimport Problem


@cython.final
cdef class Population:
    cdef public list agents
    cdef public Problem problem
    cdef public object generator
    cdef public int threads
    cdef Py_ssize_t _size
    cdef bint _minimize

    cpdef Population spawn(self, object agents)
    cpdef Population empty(self)
    cpdef Population snapshot(self)
    cpdef Population copy(self)
    cpdef Py_ssize_t size(self)
    cpdef list toarray(self)
    cpdef object evaluate(self, object agents, str mode)
    cpdef void append(self, object agent)
    cpdef Population sort(self)
    cpdef Population greedy(self, object candidates, str mode=*)
    cpdef list top(self, Py_ssize_t k)
    cpdef list bottom(self, Py_ssize_t k)
    cdef str sense_str(self)
    cdef object _batch_matrix(self, Py_ssize_t k)


cpdef Population population(object size, object range=*, type cls=*)
cpdef Population empty_snapshot(Population population)
cpdef Population snapshot(Population population)
cpdef object correct_solution(object problem, object solution)
cpdef object reset_solution(object problem, object generator, object solution)
cpdef object opposite_solution(object problem, object generator, Agent agent, Agent g_best)
