#!/usr/bin/env python
# Created by "Thieu" at 14:01, 16/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.swarm_based.FOA.OriginalFOA cimport OriginalFOA


cdef class WhaleFOA(OriginalFOA):
    """
    The original version of: Whale Fruit-fly Optimization Algorithm (WFOA)

    Links:
        1. https://doi.org/10.1016/j.eswa.2020.113502

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
    >>> model = FOA.WhaleFOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Fan, Y., Wang, P., Heidari, A.A., Wang, M., Zhao, X., Chen, H. and Li, C., 2020. Boosted hunting-based
    fruit fly optimization and advances in real-world problems. Expert Systems with Applications, 159, p.113502.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, **kwargs)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        a = 2.0 - 2.0 * epoch / self.epoch  # linearly decreased from 2 to 0
        pop_new = []
        for idx, agent in enumerate(self.population.toarray()):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.random() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * self.g_best.solution - agent.solution)
                    x = self.g_best.solution - A * D
                else:
                    # select random 1 position in pop
                    x_rand = self.population[self.generator.integers(pop_size)]
                    D = np.abs(C * x_rand.solution - agent.solution)
                    x = x_rand.solution - A * D
            else:
                D1 = np.abs(self.g_best.solution - agent.solution)
                x = (
                    D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + self.g_best.solution
                )
            smell = self.norm_consecutive_adjacent__(x)
            x = cy.correct_solution(self.problem, smell)
            child = self.population.create_agent(x)
            pop_new.append(child)
            if self.mode == "sequential":
                # the classic code evaluates x, not the child's smell vector
                child.update_solution(self.population.evaluate_solution(x), child.solution)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.spawn(cy.greedy_agents(pop_new, self.population, self.problem.sense))
