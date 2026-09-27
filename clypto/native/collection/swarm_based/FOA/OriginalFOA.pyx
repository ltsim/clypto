#!/usr/bin/env python
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



def norm_consecutive_adjacent(position):
    """Smell of a position: the norm of every pair of consecutive coordinates (cyclic)."""
    return np.array(
        [np.linalg.norm([position[x], position[x + 1]]) for x in range(0, len(position) - 1)]
        + [np.linalg.norm([position[-1], position[0]])]
    )


cdef class OriginalFOAPopulation(cy.Population):
    """Agents of :class:`OriginalFOA`: a fly starts at the smell of a random position."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        return cy.Agent(solution=norm_consecutive_adjacent(solution))


cdef class OriginalFOA(cy.Optimizer):
    """
    The original version of: Fruit-fly Optimization Algorithm (FOA)

    Links:
        1. https://doi.org/10.1016/j.knosys.2011.07.001

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import FOA    >>> import numpy as np
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
    >>> model = FOA.OriginalFOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Pan, W.T., 2012. A new fruit fly optimization algorithm: taking the financial distress model
    as an example. Knowledge-Based Systems, 26, pp.69-74.
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
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalFOAPopulation)

    def norm_consecutive_adjacent__(self, position=None):
        return norm_consecutive_adjacent(position)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx, agent in enumerate(self.population.toarray()):
            x = self.population[
                idx
            ].solution + self.generator.random() * self.generator.normal(
                self.problem.bounds.low, self.problem.bounds.up
            )
            x = self.norm_consecutive_adjacent__(x)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            pop_new.append(child)
            if self.mode == "sequential":
                # the classic code evaluates x, not the child's smell vector
                child.update_solution(self.population.evaluate_solution(x), child.solution)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.spawn(cy.greedy_agents(pop_new, self.population, self.problem.sense))
