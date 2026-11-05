import cython

cimport numpy as cnp

from clypto.optimizer.native.problem cimport Problem


@cython.final
cdef class Agent:
    cdef public cnp.ndarray solution
    cdef readonly cnp.ndarray objectives
    cdef readonly object weights
    cdef readonly double fitness

    # -- state of algorithms (all None/0 until the algorithm sets them) --------
    cdef public object acc
    cdef public object age
    cdef public object bitstring
    cdef public object cost
    cdef public object crossover
    cdef public object damage
    cdef public object den
    cdef public object flag
    cdef public object g
    cdef public object hunger
    cdef public object intensity
    cdef public object interaction
    cdef public object local_best
    cdef public object local_solution
    cdef public object local_vector
    cdef public object loudness
    cdef public object m
    cdef public object mask
    cdef public object mass
    cdef public object model_weights
    cdef public object mu
    cdef public object mutation
    cdef public object nutrients
    cdef public cnp.ndarray pbest_solution
    cdef public double pbest_fitness
    cdef public object pop_size
    cdef public object pulse_frequency
    cdef public object pulse_rate
    cdef public object q_table
    cdef public object sigma
    cdef public object solution_new
    cdef public object step
    cdef public object strategy
    cdef public object target_move
    cdef public object target_new
    cdef public object target_solution
    cdef public object v
    cdef public object velocity
    cdef public object vol
    cdef public object weight
    cdef public object win

    cdef void set_evaluation(self, object objectives, object weights)
    cdef void copy_evaluation(self, Agent other)
    cpdef Agent evaluate(self, Problem problem)
    cpdef void update_solution(self, Agent other, object solution=*)


cpdef Agent duplicate_agent(Agent agent)
cpdef bint sync_if_duplicate(Agent that, Agent other)
cpdef bint better_fitness(double x, double y, str sense)
cpdef bint is_better(Agent x, Agent y, str sense)
cpdef Agent get_better_agent(Agent x, Agent y, str sense=*, bint reverse=*)
cpdef list argsort_agents(object agents, str sense)
cpdef list sort_agents(object agents, str sense)
cpdef list greedy_agents(object old, object new, str sense, str mode=*)
