#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalTWOAgent(cy.Agent):
    cdef public object weight


cdef class OriginalTWOPopulation(cy.Population):
    """Agents of :class:`OriginalTWO`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        """
        Generate new agent with solution

        Args:
            solution (np.ndarray): The solution
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return OriginalTWOAgent(solution=solution, weight=0.0)


cdef class OriginalTWO(cy.Optimizer):
    """
    The original version of: Tug of War Optimization (TWO)

    Links:
        1. https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.OriginalTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for
    optimal design of structures (pp. 451-487). Springer, Cham.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalTWOPopulation)
        self.muy_s = 1
        self.muy_k = 1
        self.delta_t = 1
        self.alpha = 0.99
        self.beta = 0.1

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.population = self.update_weight__(self.population)

    def update_weight__(self, teams):
        pop_size = self.population.size()
        list_fits = np.array([agent.fitness for agent in teams])
        maxx, minn = np.max(list_fits), np.min(list_fits)
        if maxx == minn:
            list_fits = self.generator.uniform(0.0, 1.0, pop_size)
        list_weights = np.exp(-(list_fits - maxx) / (maxx - minn))
        list_weights = list_weights / np.sum(list_weights) + 0.1
        for idx in range(pop_size):
            teams[idx].weight = list_weights[idx]
        return teams

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = self.population.copy()
        for idx in range(pop_size):
            pos_new = pop_new[idx].solution.copy().astype(float)
            for jdx in range(pop_size):
                if self.population[idx].weight < self.population[jdx].weight:
                    force = max(
                        self.population[idx].weight * self.muy_s,
                        self.population[jdx].weight * self.muy_s,
                    )
                    resultant_force = force - self.population[idx].weight * self.muy_k
                    g = self.population[jdx].solution - self.population[idx].solution
                    acceleration = (
                        resultant_force * g / (self.population[idx].weight * self.muy_k)
                    )
                    delta_x = 0.5 * acceleration + np.power(
                        self.alpha, epoch
                    ) * self.beta * (
                        self.problem.bounds.up - self.problem.bounds.low
                    ) * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                    pos_new += delta_x
            pop_new[idx].solution = pos_new
        for idx in range(pop_size):
            pos_new = pop_new[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if (
                    pos_new[jdx] < self.problem.bounds.low[jdx]
                    or pos_new[jdx] > self.problem.bounds.up[jdx]
                ):
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = self.g_best.solution[
                            jdx
                        ] + self.generator.standard_normal() / epoch * (
                            self.g_best.solution[jdx] - pos_new[jdx]
                        )
                        if (
                            pos_new[jdx] < self.problem.bounds.low[jdx]
                            or pos_new[jdx] > self.problem.bounds.up[jdx]
                        ):
                            pos_new[jdx] = self.population[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < self.problem.bounds.low[jdx]:
                            pos_new[jdx] = self.problem.bounds.low[jdx]
                        if pos_new[jdx] > self.problem.bounds.up[jdx]:
                            pos_new[jdx] = self.problem.bounds.up[jdx]
            pos_new = self.population.correct_solution(pos_new)
            pop_new[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[idx].evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(pop_new[idx], self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        self.population = self.update_weight__(self.population)
