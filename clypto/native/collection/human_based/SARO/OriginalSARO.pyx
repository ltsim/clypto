#!/usr/bin/env python
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.human_based.SARO.DevSARO cimport DevSARO
cimport clypto.core as cy


cdef class OriginalSARO(DevSARO):
    """
    The original version of: Search And Rescue Optimization (SARO)

    Links:
       1. https://doi.org/10.1155/2019/2482543

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): [10, 20], maximum unsuccessful search number, default = 15

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import SARO    >>> import numpy as np
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
    >>> model = SARO.OriginalSARO(epoch=1000, pop_size=50, se = 0.5, mu = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Shabani, A., Asgarian, B., Gharebaghi, S.A., Salido, M.A. and Giret, A., 2019. A new optimization
    algorithm based on search and rescue operations. Mathematical Problems in Engineering, 2019.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            se: float = 0.5,
            mu: int = 10,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 15
        """
        super().__init__(epoch, pop_size, se, mu, **kwargs)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_x = [cy.duplicate_agent(agent) for agent in self.population[: pop_size]]
        pop_m = [cy.duplicate_agent(agent) for agent in self.population[pop_size:]]
        pop_new = []
        for idx in range(pop_size):
            ## Social Phase
            k = self.generator.choice(list(set(range(0, 2 * pop_size)) - {idx}))
            sd = pop_x[idx].solution - self.population[k].solution
            j_rand = self.generator.integers(0, self.problem.n_dims)
            r1 = self.generator.uniform(-1, 1)

            x = pop_x[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                if self.generator.uniform() < self.se or j == j_rand:
                    if cy.is_better(self.population[k], pop_x[idx], self.problem.sense):
                        x[j] = self.population[k].solution[j] + r1 * sd[j]
                    else:
                        x[j] = pop_x[idx].solution[j] + r1 * sd[j]
                if x[j] < self.problem.bounds.low[j]:
                    x[j] = (pop_x[idx].solution[j] + self.problem.bounds.low[j]) / 2
                if x[j] > self.problem.bounds.up[j]:
                    x[j] = (pop_x[idx].solution[j] + self.problem.bounds.up[j]) / 2
            x = cy.reset_solution(self.problem, self.generator, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
        pop_new = self.population.evaluate(pop_new, self.mode)
        for idx in range(0, pop_size):
            if cy.is_better(pop_new[idx], pop_x[idx], self.problem.sense):
                pop_m[self.generator.integers(0, pop_size)] = cy.duplicate_agent(pop_x[idx])
                pop_x[idx] = cy.duplicate_agent(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        ## Individual phase
        pop = pop_x.copy() + pop_m.copy()
        pop_new = []
        for idx in range(0, pop_size):
            k, m = self.generator.choice(
                list(set(range(0, 2 * pop_size)) - {idx}), 2, replace=False
            )
            x = pop_x[idx].solution + self.generator.uniform() * (
                    pop[k].solution - pop[m].solution
            )
            for j in range(0, self.problem.n_dims):
                if x[j] < self.problem.bounds.low[j]:
                    x[j] = (pop_x[idx].solution[j] + self.problem.bounds.low[j]) / 2
                if x[j] > self.problem.bounds.up[j]:
                    x[j] = (pop_x[idx].solution[j] + self.problem.bounds.up[j]) / 2
            x = cy.reset_solution(self.problem, self.generator, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
        pop_new = self.population.evaluate(pop_new, self.mode)
        for idx in range(0, pop_size):
            if cy.is_better(pop_new[idx], pop_x[idx], self.problem.sense):
                pop_m[self.generator.integers(0, pop_size)] = pop_x[idx]
                pop_x[idx] = cy.duplicate_agent(pop_new[idx])
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.population.generate_agent()
                self.dyn_USN[idx] = 0
        self.population = self.population.spawn(pop_x + pop_m)
