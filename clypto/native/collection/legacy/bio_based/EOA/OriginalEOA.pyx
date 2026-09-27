#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalEOA(cy.Optimizer):
    """
    The developed version: Earthworm Optimisation Algorithm (EOA)

    Links:
        1. http://doi.org/10.1504/IJBIC.2015.10004283
        2. https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa

    Notes:
        The original version from matlab code above will not work well, even with small dimensions.
        I change updating process, change cauchy process using x_mean, use global best solution

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_c (float): (0, 1) -> better [0.5, 0.95], crossover probability
        + p_m (float): (0, 1) -> better [0.01, 0.2], initial mutation probability
        + n_best (int): (2, pop_size/2) -> better [2, 5], how many of the best earthworm to keep from one generation to the next
        + alpha (float): (0, 1) -> better [0.8, 0.99], similarity factor
        + beta (float): (0, 1) -> better [0.8, 1.0], the initial proportional factor
        + gama (float): (0, 1) -> better [0.8, 0.99], a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import EOA    >>> import numpy as np
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
    >>> model = EOA.OriginalEOA(epoch=1000, pop_size=50, p_c = 0.9, p_m = 0.01, n_best = 2, alpha = 0.98, beta = 0.9, gama = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    cdef public double alpha
    cdef public double beta
    cdef public double gama
    cdef public int n_best
    cdef public double p_c
    cdef public double p_m

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p_c: float = 0.9,
            p_m: float = 0.01,
            n_best: int = 2,
            alpha: float = 0.98,
            beta: float = 0.9,
            gama: float = 0.9,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_c (float): default = 0.9, crossover probability
            p_m (float): default = 0.01 initial mutation probability
            n_best (int): default = 2, how many of the best earthworm to keep from one generation to the next
            alpha (float): default = 0.98, similarity factor
            beta (float): default = 0.9, the initial proportional factor
            gama (float): default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.
        """
        super().__init__(parameters=["epoch", "pop_size", "p_c", "p_m", "n_best", "alpha", "beta", "gama"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.p_c = cy.validator(float, p_c, (0, 1.0), "p_c")
        self.p_m = cy.validator(float, p_m, (0, 1.0), "p_m")
        self.n_best = cy.validator(int, n_best, [2, int(self.population.size() / 2)], "n_best")
        self.alpha = cy.validator(float, alpha, (0, 1.0), "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")
        self.gama = cy.validator(float, gama, (0, 1.0), "gama")

    def initialize_variables(self):
        self.dyn_beta = self.beta

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Update the pop best
        pop_elites = self.population.sort()
        pop = []
        for idx in range(0, pop_size):
            ### Reproduction 1: the first way of reproducing
            x_t1 = (
                    self.problem.bounds.low + self.problem.bounds.up - self.alpha * self.population[idx].solution
            )

            ### Reproduction 2: the second way of reproducing
            if (
                    idx >= self.n_best
            ):  ### Select two parents to mate and create two children
                idx = int(pop_size * 0.2)
                if (
                        self.generator.uniform() < 0.5
                ):  ## 80% parents selected from best population
                    idx1, idx2 = self.generator.choice(range(0, idx), 2, replace=False)
                else:  ## 20% left parents selected from worst population (make more diversity)
                    idx1, idx2 = self.generator.choice(
                        range(idx, pop_size), 2, replace=False
                    )
                r = self.generator.uniform()
                x_child = (
                        r * self.population[idx2].solution + (1 - r) * self.population[idx1].solution
                )
            else:
                r1 = self.generator.integers(0, pop_size)
                x_child = self.population[r1].solution
            x_t1 = self.dyn_beta * x_t1 + (1.0 - self.dyn_beta) * x_child
            pos_new = self.population.correct_solution(x_t1)
            agent = self.population.create_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop = self.population.evaluate(pop, self.mode)
            self.population = self.population.greedy(pop)
        self.dyn_beta = self.gama * self.beta
        self.population = self.population.sort()[:pop_size]

        pos_list = np.array([agent.solution for agent in self.population])
        x_mean = np.mean(pos_list, axis=0)
        ## Cauchy mutation (CM)
        cauchy_w = self.g_best.solution.copy()
        pop_new = []
        for idx in range(
                self.n_best, pop_size
        ):  # Don't allow the elites to be mutated
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            cauchy_w = np.where(condition, x_mean, cauchy_w)
            x_t1 = (cauchy_w + self.g_best.solution) / 2
            pos_new = self.population.correct_solution(x_t1)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population[self.n_best:] = cy.greedy_agents(pop_new, self.population[self.n_best:], self.problem.sense)

        ## Elitism Strategy: Replace the worst with the previous generation's elites.
        self.population = self.population.sort()
        for idx in range(0, self.n_best):
            self.population[pop_size - idx - 1] = pop_elites[idx].copy()

        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, agent in enumerate(self.population):
            if tuple(agent.solution.tolist()) in new_set:
                self.population[idx] = self.population.generate_agent()
            else:
                new_set.add(tuple(agent.solution.tolist()))
