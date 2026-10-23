#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""``Agent``: a solution and its evaluation, plus the functions that compare and copy agents.

An agent holds its ``solution`` and, once evaluated, its ``objectives``, their
``weights`` and the weighted ``fitness``. The evaluation is read-only: it changes
only through ``evaluate(problem)`` or ``update_solution(other)``. Algorithms that
need extra per-agent state subclass ``Agent``, declare typed ``cdef public`` fields
and override ``clone`` to copy them: copies are static C code, never ``setattr``.
"""
import numbers

import numpy as np
cimport numpy as cnp

from clypto.optimizer.native.problem cimport Problem

cnp.import_array()

cdef tuple SUPPORTED_ARRAY = (tuple, list, np.ndarray)
cpdef Agent duplicate_agent(Agent agent):
    """Copy of ``agent`` (``agent.clone()``): its fields and evaluation are shared (replaced, never mutated)."""
    return agent.clone()


cpdef bint sync_if_duplicate(Agent that, Agent other):
    """Give ``that`` the evaluation of ``other`` when their solutions match (within 1e-6)."""
    if that == other:
        that.copy_evaluation(other)
        return True
    return False


cpdef bint better_fitness(double x, double y, str sense):
    """True when fitness ``x`` is better than ``y``: ``x < y`` for min, ``not (x < y)`` for max."""
    if sense == "min":
        return x < y
    return not (x < y)


cpdef bint is_better(Agent x, Agent y, str sense):
    """True when agent ``x`` is better than ``y`` (``better_fitness`` of their fitness)."""
    return better_fitness(x.fitness, y.fitness, sense)


cpdef Agent get_better_agent(Agent x, Agent y, str sense="min", bint reverse=False):
    """Copy of the better agent; a tie keeps ``y`` when minimizing, ``x`` when maximizing."""
    cdef bint maximize = (sense != "min") != reverse
    if x.fitness < y.fitness:
        return y.clone() if maximize else x.clone()
    return x.clone() if maximize else y.clone()


cpdef list argsort_agents(object agents, str sense):
    """Positions best first (``np.argsort`` of the fitness, reversed when maximizing)."""
    order = np.argsort(np.array([agent.fitness for agent in agents], dtype=float)).tolist()
    return order[::-1] if sense == "max" else order


cpdef list greedy_agents(object old, object new, str sense, str mode="swarm"):
    """Per position, the ``new`` agent when better, else the ``old`` one (lists of equal length).

    The batch modes keep ``new`` only when strictly better. ``mode="sequential"`` applies the
    one-by-one rule of ``get_better_agent`` (``better_fitness``: when maximizing, a tie keeps ``new``),
    so an algorithm that evaluates its whole batch at once keeps its sequential results exactly.
    """
    if len(old) != len(new):
        raise ValueError("Greedy selection of two population with different length.")
    if mode == "sequential":
        return [n if better_fitness(n.fitness, o.fitness, sense) else o for o, n in zip(old, new)]
    if sense == "max":
        return [n if n.fitness > o.fitness else o for o, n in zip(old, new)]
    return [n if n.fitness < o.fitness else o for o, n in zip(old, new)]


cpdef list sort_agents(object agents, str sense):
    """The agents best first (see ``argsort_agents``)."""
    return [agents[i] for i in argsort_agents(agents, sense)]


cdef class Agent:
    def __cinit__(self):
        self.fitness = np.nan

    def __init__(self, solution=None, objectives=None, weights=None):
        self.solution = solution
        if objectives is not None:
            self.set_evaluation(objectives, weights)

    cdef Agent clone(self):
        """A new agent of the same class sharing ``solution`` and the evaluation.

        Subclasses with fields override it: ``new = <XAgent>Agent.clone(self)``, then copy each field.
        """
        cdef Agent new = <Agent>type(self).__new__(type(self))
        new.solution = self.solution
        new.copy_evaluation(self)
        return new

    cdef void set_evaluation(self, object objectives, object weights):
        """Store ``objectives`` (number or sequence) and ``weights``; fitness is their dot product."""
        if type(objectives) not in SUPPORTED_ARRAY:
            if not isinstance(objectives, numbers.Number):
                raise ValueError("Invalid objectives. It should be a list, tuple, np.ndarray, int or float.")
            objectives = [objectives]
        self.objectives = np.array(objectives).flatten()
        cdef Py_ssize_t n = len(self.objectives)
        if weights is None:
            self.weights = n
        elif type(weights) in SUPPORTED_ARRAY:
            self.weights = np.array(weights).flatten()
        elif isinstance(weights, numbers.Number):
            self.weights = np.array([weights] * n).flatten()
        else:
            raise ValueError("Invalid weights. It should be a list, tuple, np.ndarray.")
        fitness_weights = self.weights
        if not (type(fitness_weights) in SUPPORTED_ARRAY and len(fitness_weights) == n):
            fitness_weights = n * (1.0,)
        self.fitness = np.dot(fitness_weights, self.objectives)

    cdef void copy_evaluation(self, Agent other):
        self.objectives = other.objectives
        self.weights = other.weights
        self.fitness = other.fitness

    def evaluate(self, Problem problem):
        """Evaluate ``solution`` on ``problem`` (counted in ``problem.n_evals``)."""
        self.set_evaluation(problem.evaluate_solution(self.solution), problem._obj_weights)
        return self

    def update_solution(self, Agent other, solution=None):
        """Take the evaluation of ``other`` and move to ``solution`` (default: a copy of ``other.solution``).

        ``solution`` is stored as given, so ``agent.update_solution(candidate, pos_new)``
        keeps ``pos_new`` itself; ``agent.update_solution(other, agent.solution)``
        only takes the evaluation.
        """
        self.solution = other.solution.copy() if solution is None else solution
        self.copy_evaluation(other)

    def __repr__(self):
        return f"{type(self).__name__}(fitness={self.fitness}, solution={self.solution})"

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self):
        if self.objectives is None:
            raise ValueError("Agent cannot generate a value from fitness")
        return self.fitness
