#!/usr/bin/env python
# Created by "Thieu" at 08:58, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Legacy engine: the population is a :class:`Population` of agents evolved one at a time.

This is the base of the legacy collection (``cy.Optimizer``). An algorithm
declares its hyper-parameters in ``__init__``::

    super().__init__(parameters=["epoch", "pop_size", "c1"], sort_flag=False, **kwargs)
    self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
    self.population = cy.population(pop_size, range=[5, 10000])
    self.c1 = cy.validator(float, c1, (0, 5.0), "c1")

and implements ``evolve(epoch)``. ``solve()`` binds the population to the problem
and fills it; a plain list assigned to ``self.population`` is wrapped in a
population of the same class.
"""
from clypto.optimizer.native.population cimport Population
from clypto.optimizer.native.problem import Problem


cdef class LegacyOptimizer(NativeOptimizer):
    def __init__(self, parameters=(), sort_flag=False, **kwargs):
        NativeOptimizer.__init__(self, parameters, sort_flag, kwargs.get("name"), kwargs.get("mode"))
        self._population = None
        self.g_best = None
        self.g_worst = None
        self.problem = None

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        if value is None or isinstance(value, Population):
            self._population = value
        else:
            self._population = self._population.spawn(value)

    @property
    def pop_size(self):
        """The configured population size (``population.size()``)."""
        return self._population.size()

    @pop_size.setter
    def pop_size(self, value):
        self._population.resize(value)

    # -- engine steps --------------------------------------------------------------
    cdef void check_problem(self, object problem, object seed):
        if self._population is None:
            raise ValueError(f"{type(self).__name__} must declare self.population = cy.population(pop_size) in __init__.")
        self.problem = Problem.coerce(problem, seed)
        self.g_best, self.g_worst = None, None
        self._population.bind(self.problem, self.generator)

    cdef void before_initialization(self, object starting_solutions):
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
        self.population = self._population.generate(starting=starting_solutions)

    def initialization(self):
        if len(self._population) == 0:
            self.population = self._population.generate()

    cdef void after_initialization(self):
        # The initial population is sorted or not depending on the algorithm;
        # g_best/g_worst start as copies.
        ranked = self._population.sort()
        self.g_best, self.g_worst = ranked[0].copy(), ranked[-1].copy()
        if self.sort_flag:
            self.population = ranked

    cdef void after_evolve(self):
        # g_best is the best agent of the population itself (an alias, not a copy).
        ranked = self._population.sort()
        self.g_best = ranked[0]
        if self.sort_flag:
            self.population = ranked
