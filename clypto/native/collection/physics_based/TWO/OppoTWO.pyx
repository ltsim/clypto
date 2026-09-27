#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.physics_based.TWO.OriginalTWO cimport OriginalTWO


cdef class OppoTWO(OriginalTWO):
    """
    The opossition-based learning version: Tug of War Optimization (OTWO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.OppoTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
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

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        half_size = -(-pop_size // 2)  # ceil division, safe for odd pop_size
        list_idx = self.generator.choice(
            range(0, pop_size), half_size, replace=False
        )
        pop_temp = [self.population[list_idx[idx]] for idx in range(0, half_size)]
        pop_oppo = []
        for idx in range(len(pop_temp)):
            pos_opposite = self.problem.bounds.up + self.problem.bounds.low - pop_temp[idx].solution
            pos_opposite = cy.correct_solution(self.problem, pos_opposite)
            agent = self.population.create_agent(pos_opposite)
            pop_oppo.append(agent)
        pop_oppo = self.population.evaluate(pop_oppo, self.mode)
        self.population = self.population.spawn((pop_temp + pop_oppo)[: pop_size])
        self.population = self.population.spawn(self.update_weight__(self.population))

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Apply force of others solution on each individual solution
        pop_new = self.population.copy()
        for idx, agent in enumerate(self.population.toarray()):
            x = pop_new[idx].solution.copy().astype(float)
            for jdx in range(pop_size):
                if agent.weight < self.population[jdx].weight:
                    force = max(
                        agent.weight * self.muy_s,
                        self.population[jdx].weight * self.muy_s,
                    )
                    resultant_force = force - agent.weight * self.muy_k
                    g = self.population[jdx].solution - agent.solution
                    acceleration = (
                        resultant_force * g / (agent.weight * self.muy_k)
                    )
                    delta_x = 1 / 2 * acceleration + np.power(
                        self.alpha, epoch
                    ) * self.beta * (
                        self.problem.bounds.up - self.problem.bounds.low
                    ) * self.generator.normal(
                        0, 1, self.problem.n_dims
                    )
                    x += delta_x
            agent.solution = x
        ## Amend solution and update fitness value
        for idx, agent in enumerate(self.population.toarray()):
            x = self.g_best.solution + self.generator.normal(
                0, 1, self.problem.n_dims
            ) / (epoch) * (self.g_best.solution - pop_new[idx].solution)
            conditions = np.logical_or(
                pop_new[idx].solution < self.problem.bounds.low,
                pop_new[idx].solution > self.problem.bounds.up,
            )
            conditions = np.logical_and(
                conditions, self.generator.random(self.problem.n_dims) < 0.5
            )
            x = np.where(conditions, x, agent.solution)
            pop_new[idx].solution = cy.correct_solution(self.problem, x)
            if self.mode == "sequential":
                # the classic code evaluates x, not pop_new[idx].solution (MEALPY behaviour, kept)
                pop_new[idx].update_solution(self.population.evaluate_solution(x), pop_new[idx].solution)
                self.population[idx] = cy.get_better_agent(pop_new[idx], self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ## Opposition-based here
        pop = []
        for idx, agent in enumerate(self.population.toarray()):
            C_op = cy.opposite_solution(self.problem, self.generator, agent, self.g_best)
            x = cy.correct_solution(self.problem, C_op)
            child = self.population.create_agent(x)
            pop.append(child)
        self.population = self.population.greedy(self.population.evaluate(pop, self.mode), self.mode)
        self.population = self.population.spawn(self.update_weight__(self.population))
