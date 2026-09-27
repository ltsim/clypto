#!/usr/bin/env python
# Created by "Thieu" at 08:58, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""The engine of the collection (``cy.Optimizer``): the population is a :class:`Population` of agents.

An algorithm declares its hyper-parameters in ``__init__``::

    super().__init__(parameters=["epoch", "pop_size", "c1"], sort_flag=False, **kwargs)
    self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
    self.population = cy.population(pop_size, range=[5, 10000])
    self.c1 = cy.validator(float, c1, (0, 5.0), "c1")

and implements ``cdef void evolve(self, int epoch)``. ``solve()`` binds the
population to the problem and fills it. ``self.population`` is a typed C field:
a list of agents is assigned as ``self.population.spawn(agents)``.
"""
from clypto.optimizer.native.agent cimport duplicate_agent
from clypto.optimizer.native.population cimport Population
from clypto.optimizer.native.problem import Problem


cdef class LegacyOptimizer(NativeOptimizer):
    def __init__(self, parameters=(), sort_flag=False, **kwargs):
        NativeOptimizer.__init__(self, parameters, sort_flag, kwargs.get("name"), kwargs.get("mode"))
        self.population = None
        self.g_best = None
        self.g_worst = None
        self.problem = None

    @property
    def pop_size(self):
        """The configured population size (``population.size()``)."""
        return self.population.size()

    @pop_size.setter
    def pop_size(self, value):
        self.population.resize(value)

    # -- engine steps --------------------------------------------------------------
    cdef void check_problem(self, object problem, object seed):
        if self.population is None:
            raise ValueError(f"{type(self).__name__} must declare self.population = cy.population(pop_size) in __init__.")
        self.problem = Problem.coerce(problem, seed)
        self.g_best, self.g_worst = None, None
        self.population.bind(self.problem, self.generator)

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
        self.population = self.population.generate(starting=starting_solutions)

    def initialization(self):
        if len(self.population) == 0:
            self.population = self.population.generate()

    cdef void after_initialization(self):
        # The initial population is sorted or not depending on the algorithm;
        # g_best/g_worst start as copies.
        ranked = self.population.sort()
        self.g_best, self.g_worst = duplicate_agent(ranked[0]), duplicate_agent(ranked[-1])
        if self.sort_flag:
            self.population = ranked

    cdef void after_evolve(self):
        # g_best is the best agent of the population itself (an alias, not a copy).
        ranked = self.population.sort()
        self.g_best = ranked[0]
        if self.sort_flag:
            self.population = ranked
