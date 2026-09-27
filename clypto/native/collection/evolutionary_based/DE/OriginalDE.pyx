#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport numpy as cnp
cimport clypto.core as cy


cdef class OriginalDE(cy.Optimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wf (float): [-1., 1.0], weighting factor, default = 0.1
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import DE    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = DE.OriginalDE(epoch=1000, pop_size=50, wf = 0.7, cr = 0.9, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    cdef public double cr
    cdef public int strategy
    cdef public double wf

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wf: float = 0.1,
        cr: float = 0.9,
        strategy: int = 0,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        super().__init__(parameters=["epoch", "pop_size", "wf", "cr", "strategy"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.wf = cy.validator(float, wf, (-3.0, 3.0), "wf")
        self.cr = cy.validator(float, cr, (0, 1.0), "cr")
        self.strategy = cy.validator(int, strategy, [0, 5], "strategy")

    # -- strategies: the trial vector of agent ``idx`` (the draws of each block, in the same order) ----------
    cdef cnp.ndarray strategy_0(self, int idx, int pop_size):
        """DE/rand/1: three random agents other than ``idx``."""
        r = self.generator.choice(list(set(range(0, pop_size)) - {idx}), 3, replace=False)
        return self.population[r[0]].solution + self.wf * (self.population[r[1]].solution - self.population[r[2]].solution)

    cdef cnp.ndarray strategy_1(self, int idx, int pop_size):
        """DE/best/1."""
        r = self.generator.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)
        return self.g_best.solution + self.wf * (self.population[r[0]].solution - self.population[r[1]].solution)

    cdef cnp.ndarray strategy_2(self, int idx, int pop_size):
        """DE/best/2."""
        r = self.generator.choice(list(set(range(0, pop_size)) - {idx}), 4, replace=False)
        return (
            self.g_best.solution
            + self.wf * (self.population[r[0]].solution - self.population[r[1]].solution)
            + self.wf * (self.population[r[2]].solution - self.population[r[3]].solution)
        )

    cdef cnp.ndarray strategy_3(self, int idx, int pop_size):
        """DE/rand/2."""
        r = self.generator.choice(list(set(range(0, pop_size)) - {idx}), 5, replace=False)
        return (
            self.population[r[0]].solution
            + self.wf * (self.population[r[1]].solution - self.population[r[2]].solution)
            + self.wf * (self.population[r[3]].solution - self.population[r[4]].solution)
        )

    cdef cnp.ndarray strategy_4(self, int idx, int pop_size):
        """DE/current-to-best/1."""
        r = self.generator.choice(list(set(range(0, pop_size)) - {idx}), 2, replace=False)
        current = self.population[idx].solution
        return (
            current
            + self.wf * (self.g_best.solution - current)
            + self.wf * (self.population[r[0]].solution - self.population[r[1]].solution)
        )

    cdef cnp.ndarray strategy_5(self, int idx, int pop_size):
        """DE/current-to-rand/1."""
        r = self.generator.choice(list(set(range(0, pop_size)) - {idx}), 3, replace=False)
        current = self.population[idx].solution
        return (
            current
            + self.wf * (self.population[r[0]].solution - current)
            + self.wf * (self.population[r[1]].solution - self.population[r[2]].solution)
        )

    cdef cnp.ndarray trial(self, int idx, int pop_size):
        """The trial vector of the configured ``strategy``."""
        if self.strategy == 0:
            return self.strategy_0(idx, pop_size)
        if self.strategy == 1:
            return self.strategy_1(idx, pop_size)
        if self.strategy == 2:
            return self.strategy_2(idx, pop_size)
        if self.strategy == 3:
            return self.strategy_3(idx, pop_size)
        if self.strategy == 4:
            return self.strategy_4(idx, pop_size)
        return self.strategy_5(idx, pop_size)

    cdef cnp.ndarray mutation(self, cnp.ndarray current, cnp.ndarray x):
        """Binomial crossover of ``x`` into ``current`` (rate ``cr``), bounded."""
        condition = self.generator.random(self.problem.n_dims) < self.cr
        return cy.correct_solution(self.problem, np.where(condition, x, current))

    cdef void select(self, int idx, cy.Agent child):
        """Sequential selection: evaluate now and keep the better of ``child`` and the current agent."""
        child.evaluate(self.problem)
        self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm: strategy -> mutation -> selection.

        Args:
            epoch (int): The current iteration
        """
        cdef int pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        cdef cy.Agent agent, child
        for idx, agent in enumerate(self.population.toarray()):
            child = self.population.create_agent(self.mutation(agent.solution, self.trial(idx, pop_size)))
            n_population.append(child)
            if self.mode == "sequential":
                # order-dependent: the next trials read the agents already replaced
                self.select(idx, child)
        if self.mode != "sequential":
            self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
