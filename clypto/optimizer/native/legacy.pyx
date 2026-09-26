#!/usr/bin/env python
# Created by "Thieu" at 08:58, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Legacy engine: the population is a list of agent objects evolved one at a time.

This is the base of the legacy collection and of ``@cy.legacy`` classes. Subclasses
set their hyper-parameters in ``__init__`` (register them with ``_set_parameters``),
implement ``_evolve(epoch)`` and may override the other ``_`` hooks.
"""
import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.population import Population
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.native.target import NativeTarget
from clypto.optimizer.validator import Validator


cdef class LegacyOptimizer(NativeOptimizer):
    def __init__(self, **kwargs):
        NativeOptimizer.__init__(self, name=kwargs.get("name"), mode=kwargs.get("mode"))
        self.validator = Validator()
        self.pop = None
        self.g_best = LegacyAgent()
        self.g_worst = None
        self.problem = None

    # -- lifecycle hooks -----------------------------------------------------------
    def _check_problem(self, problem, seed):
        self.problem = Problem.coerce(problem, seed)
        self.pop, self.g_best, self.g_worst = None, None, None

    def _before_initialization(self, starting_solutions=None):
        if starting_solutions is None:
            return
        if not (type(starting_solutions) in self.SUPPORTED_ARRAYS and len(starting_solutions) == self.pop_size):
            raise ValueError(
                "Invalid starting_solutions. It should be a list/2D matrix of positions with same length as pop_size."
            )
        if not (type(starting_solutions[0]) in self.SUPPORTED_ARRAYS and len(starting_solutions[0]) == self.problem.n_dims):
            raise ValueError(
                "Invalid starting_solutions. It should be a list of positions or 2D matrix of positions only."
            )
        self.pop = [self._generate_agent(solution) for solution in starting_solutions]

    def _initialization(self):
        if self.pop is None:
            self.pop = self._generate_population(self.pop_size)

    def _after_initialization(self):
        # The initial population is sorted or not depending on the algorithm;
        # g_best/g_worst start as copies.
        self._wrap_pop()
        ranked = self.pop.sort()
        self.g_best, self.g_worst = ranked[0].copy(), ranked[-1].copy()
        if self.sort_flag:
            self.pop = ranked

    def _after_evolve(self):
        # g_best is the best agent of pop itself (an alias, not a copy).
        self._wrap_pop()
        ranked = self.pop.sort()
        self.g_best = ranked[0]
        if self.sort_flag:
            self.pop = ranked

    def _wrap_pop(self):
        # _evolve may leave a plain list (or a Population of the other sense) in self.pop.
        if not isinstance(self.pop, Population) or self.pop.sense != self.problem.sense:
            self.pop = Population(self.pop, self.problem.sense)

    # -- agents --------------------------------------------------------------------
    def _generate_empty_agent(self, solution=None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return LegacyAgent(solution=solution)

    def _generate_agent(self, solution=None):
        agent = self._generate_empty_agent(solution)
        agent.target = self._get_target(agent.solution)
        return agent

    def _generate_population(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        return Population([self._generate_agent() for _ in range(pop_size)], self.problem.sense)

    def _amend_solution(self, solution):
        return np.clip(solution, self.problem.bounds.low, self.problem.bounds.up)

    def _correct_solution(self, solution):
        return self.problem.correct_solution(self._amend_solution(solution))

    def _update_target_for_population(self, pop):
        """Evaluate every agent of ``pop`` (only in a batch ``mode``; otherwise a no-op)."""
        cdef Py_ssize_t idx, n = len(pop)
        if self.mode == "swarm":
            for idx in range(n):
                pop[idx].target = self._get_target(pop[idx].solution, counted=False)
        elif self.mode in ("parallel", "thread", "process"):
            self._evaluate_parallel(pop, n)
        else:
            return pop
        self._nfe_counter += n
        return pop

    def _evaluate_parallel(self, pop, Py_ssize_t n):
        """Re-evaluate ``pop`` in place, on OpenMP threads when the problem has a nogil evaluator.

        Without a ``Problem.evaluator`` (the default) this is the sequential Python
        evaluation, preserving the legacy results.
        """
        cdef Py_ssize_t i
        if self.problem.evaluator is None or self.problem.n_objs != 1:
            for i in range(n):
                pop[i].target = self._get_target(pop[i].solution, counted=False)
            return
        _, O = self.problem.evaluate(np.array([agent.solution for agent in pop], dtype=np.float64), True)
        for i in range(n):
            pop[i].target = NativeTarget(O[i], self.problem.obj_weights)
