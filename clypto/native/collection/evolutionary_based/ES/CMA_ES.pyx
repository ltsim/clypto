#!/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class CMA_ESAgent(cy.Agent):
    cdef public object step

    def __init__(self, solution=None, objectives=None, weights=None, step=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.step = step

    cdef cy.Agent clone(self):
        cdef CMA_ESAgent new = <CMA_ESAgent>cy.Agent.clone(self)
        new.step = self.step
        return new


cdef class CMA_ESPopulation(cy.Population):
    """Agents of :class:`CMA_ES`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        step = self.generator.multivariate_normal(
            np.zeros(self.problem.n_dims), np.eye(self.problem.n_dims)
        )
        return CMA_ESAgent(solution=solution, step=step)


cdef class CMA_ES(cy.Optimizer):
    """
    The original version of: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

    Links:
        1. https://en.wikipedia.org/wiki/CMA-ES

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import ES    >>> import numpy as np
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
    >>> model = ES.CMA_ES(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary computation, 9(2), 159-195.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=CMA_ESPopulation)

    def before_main_loop(self):
        pop_size = self.population.size()
        self.mu = int(np.round(pop_size / 2))
        self.ps = np.zeros(self.problem.n_dims)
        self.C = np.eye(self.problem.n_dims)
        self.pc = np.zeros(self.problem.n_dims)
        self.w = np.log(pop_size + 0.5) - np.log(np.arange(1, pop_size + 1))
        self.w = self.w / np.sum(self.w)
        self.mu_eff = 1.0 / np.sum(self.w**2)  # Number of effective solutions
        # Step Size Control Parameters (c_sigma and d_sigma);
        sigma0 = 0.1 * (self.problem.bounds.up - self.problem.bounds.low)
        self.cs = (self.mu_eff + 2) / (self.problem.n_dims + self.mu_eff + 5)
        self.ds = (
            1
            + self.cs
            + 2
            * np.max(np.sqrt((self.mu_eff - 1.0) / (self.problem.n_dims + 1)) - 1, 0)
        )
        self.ENN = np.sqrt(self.problem.n_dims) * (
            1 - 1.0 / (4 * self.problem.n_dims) + 1.0 / (21 * self.problem.n_dims**2)
        )
        ## Covariance Update Parameters
        self.cc = (4 + self.mu_eff / self.problem.n_dims) / (
            4 + self.problem.n_dims + 2 * self.mu_eff / self.problem.n_dims
        )
        self.c1 = 2.0 / ((self.problem.n_dims + 1.3) ** 2 + self.mu_eff)
        alpha_mu = 2
        self.cmu = min(
            1 - self.c1,
            alpha_mu
            * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((self.problem.n_dims + 2) ** 2 + alpha_mu * self.mu_eff / 2),
        )
        self.hth = (1.4 + 2 / (self.problem.n_dims + 1)) * self.ENN
        self.sigma = sigma0
        self.x_mean = np.mean([agent.solution for agent in self.population[: self.mu]], axis=0)

    def update_step__(self, pop, cc):
        pop_size = self.population.size()
        for idx in range(0, pop_size):
            pop[idx].step = self.generator.multivariate_normal(
                np.zeros(self.problem.n_dims), cc
            )
        return pop

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            x = self.x_mean + self.sigma * agent.step
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        n_population = self.population.evaluate(n_population, self.mode)
        self.population = self.population.spawn(cy.sort_agents(n_population, self.problem.sense))
        # Update MEan
        self.population = self.population.spawn(self.update_step__(self.population, self.C))
        self.x_step = np.zeros(self.problem.n_dims)
        for idx in range(0, self.mu):
            self.x_step += self.w[idx] * self.population[idx].step
        self.x_mean = self.x_mean + self.sigma * self.x_step
        # Update Step Size
        t11 = np.dot(self.x_step, np.linalg.inv(np.linalg.cholesky(self.C).T))
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mu_eff
        ) * t11
        self.sigma = (
            self.sigma
            * np.exp(self.cs / self.ds * (np.linalg.norm(self.ps) / self.ENN - 1))
            ** 0.3
        )
        # Update Covariance Matrix
        if (
            np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * epoch))
            < self.hth
        ):
            hs = 1
        else:
            hs = 0
        delta = (1 - hs) * self.cc * (2 - self.cc)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(
            self.cc * (2 - self.cc) * self.mu_eff
        ) * self.x_step
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc))
            + delta * self.C
        )
        for idx in range(0, self.mu):
            self.C = self.C + self.cmu * self.w[idx] * np.outer(
                self.population[idx].step, self.population[idx].step
            )
        # If Covariance Matrix is not Positive Defenite or Near Singular
        E, V = np.linalg.eig(self.C)
        E = np.diag(E)
        if np.any(np.diag(E) < 0):
            E[E < 0] = 0
            self.C = V * E / V
