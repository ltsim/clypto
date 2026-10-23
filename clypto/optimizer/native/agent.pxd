cimport numpy as cnp


cdef class Agent:
    cdef public cnp.ndarray solution
    cdef readonly cnp.ndarray objectives
    cdef readonly double fitness
    cdef readonly object weights

    cdef void set_evaluation(self, object objectives, object weights)
    cdef void copy_evaluation(self, Agent other)
    cdef Agent clone(self)


cpdef Agent duplicate_agent(Agent agent)
cpdef bint sync_if_duplicate(Agent that, Agent other)
cpdef bint better_fitness(double x, double y, str sense)
cpdef bint is_better(Agent x, Agent y, str sense)
cpdef Agent get_better_agent(Agent x, Agent y, str sense=*, bint reverse=*)
cpdef list argsort_agents(object agents, str sense)
cpdef list sort_agents(object agents, str sense)
cpdef list greedy_agents(object old, object new, str sense, str mode=*)
