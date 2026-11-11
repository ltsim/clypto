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
from clypto.optimizer.native.agent cimport Agent, is_better
from clypto.optimizer.native.population cimport Population, population
from clypto.optimizer.native.problem import Problem
from clypto.optimizer.native.utils cimport validator


cdef Agent snapshot_agent(Agent agent):
    """An independent copy of ``agent`` (``snapshot``): the solution vector is copied, the rest shared."""
    return agent.snapshot()


cdef class LegacyOptimizer(NativeOptimizer):
    def __init__(self, parameters=(), sort_flag=False, epoch=None, pop_size=None, **kwargs):
        NativeOptimizer.__init__(self, parameters, sort_flag, kwargs.get("name"), kwargs.get("mode"))
        self.population = None
        self.g_best = None
        self.g_worst = None
        self.problem = None
        if epoch is not None:
            self.epoch = validator(int, epoch, [1, 1000000], "epoch")
        if pop_size is not None:
            self.population = population(pop_size, range=[5, 10000])

    @property
    def bounds(self):
        """The bounds of the bound problem (None before solve())."""
        return None if self.problem is None else self.problem.bounds

    @property
    def pop_size(self):
        """The configured population size (``population.size()``)."""
        return 0 if self.population is None else self.population.size()

    @pop_size.setter
    def pop_size(self, value):
        if self.population is None:
            self.population = population(value, range=[5, 10000])
        else:
            self.population.resize(value)

    # -- engine steps --------------------------------------------------------------
    cdef void check_problem(self, object problem, object seed):
        if self.population is None:
            self.population = population(30, range=[5, 10000])
        if self.epoch <= 0:
            self.epoch = 100
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
        self.population = self.population.spawn([self.generate_agent(x) for x in starting_solutions])

    def generate(self, agent, solution):
        """Hook the algorithm may override to lay extra per-agent state."""
        return agent

    def generate_agent(self, object solution=None):
        """One initial agent, evaluated. Stateful algorithms override it to lay the per-agent
        state that used to live on their agent subclasses."""
        cdef Agent agent = self.population.generate_agent(solution)
        self.generate(agent, agent.solution)
        return agent

    def initialization(self):
        """Create the initial population: ``Population.fill`` generates through the factory."""
        self.population.fill()
        for i in range(len(self.population)):
            self.generate(self.population[i], self.population[i].solution)

    cdef void after_initialization(self):
        # The initial population is sorted or not depending on the algorithm;
        # g_best/g_worst start as independent copies.
        ranked = self.population.sort()
        self.g_best, self.g_worst = snapshot_agent(ranked[0]), snapshot_agent(ranked[-1])
        if self.sort_flag:
            self.population = ranked

    cdef void after_evolve(self):
        # g_best is the best agent *ever* seen, not the best of the current population: an
        # algorithm whose evolve() replaces the population without per-agent elitism must not
        # lose (and report) a result it already found. g_worst stays the initialization's
        # snapshot, as in mealpy-lts, which also refreshes only g_best.
        ranked = self.population.sort()
        if self.g_best is None or is_better(ranked[0], self.g_best, self.problem.sense):
            self.g_best = snapshot_agent(ranked[0])
        if self.sort_flag:
            self.population = ranked
