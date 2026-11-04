from clypto.optimizer.native.agent cimport Agent


cdef class Population:
    cdef public list agents
    cdef public object problem
    cdef public object generator
    cdef public object idx
    cdef Py_ssize_t _size

    cdef void copy_state(self, Population new)
    cpdef Population spawn(self, object agents)
    cpdef Py_ssize_t size(self)
    cpdef list toarray(self)
    cpdef object evaluate(self, object agents, object mode)
    cpdef void append(self, object agent)
    cpdef Population sort(self)
    cpdef Population greedy(self, object candidates, str mode=*)


cpdef Population population(object size, object range=*, type cls=*)
cpdef Population empty_snapshot(Population population)
cpdef Population snapshot(Population population)
cpdef object correct_solution(object problem, object solution)
cpdef object reset_solution(object problem, object generator, object solution)
cpdef object opposite_solution(object problem, object generator, Agent agent, Agent g_best)
