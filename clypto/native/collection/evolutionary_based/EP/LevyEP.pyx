#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.evolutionary_based.EP.OriginalEP cimport OriginalEP


cdef class LevyEP(OriginalEP):
    """
    The developed Levy-flight version: Evolutionary Programming (LevyEP)

    Notes:
        + Levy-flight is applied to EP, flow and some equations is changed.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import EP    >>> import numpy as np
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
    >>> model = EP.LevyEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        bout_size: float = 0.05,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        super().__init__(epoch, pop_size, bout_size, **kwargs)
        self.sort_flag = True

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        child = []
        for idx in range(0, pop_size):
            x = self.population[idx].solution + self.population[
                idx
            ].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            x = cy.correct_solution(self.problem, x)
            s_old = (
                self.population[idx].strategy
                + self.generator.normal(0, 1.0, self.problem.n_dims)
                * np.abs(self.population[idx].strategy) ** 0.5
            )
            agent = self.population.create_agent(x)
            agent.solution = x
            agent.strategy = s_old
            agent.win = 0
            child.append(agent)
        child = self.population.evaluate(child, self.mode)
        # Update the global best
        children = cy.sort_agents(child, self.problem.sense)
        pop = children + self.population
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = self.generator.integers(0, len(pop))
                if cy.is_better(pop[i], pop[rand_idx], self.problem.sense):
                    pop[i].win += 1
                else:
                    pop[rand_idx].win += 1
        ## Keep the top population, but 50% of left population will make a comeback an take the good position
        pop = sorted(pop, key=lambda agent: agent.win, reverse=True)
        pop_new = pop[: pop_size]
        pop_left = pop[pop_size :]
        ## Choice random 50% of population left
        pop_comeback = []
        idx_list = self.generator.choice(
            range(0, len(pop_left)), int(0.5 * len(pop_left)), replace=False
        )
        for idx in idx_list:
            x = pop_left[idx].solution + cy.levy_flight(self.generator, beta=1.0, multiplier=0.01, size=self.problem.n_dims, case=0)
            x = cy.correct_solution(self.problem, x)
            strategy = self.distance = 0.05 * (self.problem.bounds.up - self.problem.bounds.low)
            agent = self.population.create_agent(x)
            agent.solution = x
            agent.strategy = strategy
            agent.win = 0
            pop_comeback.append(agent)
        pop_comeback = self.population.evaluate(pop_comeback, self.mode)
        self.population = self.population.spawn(cy.sort_agents(pop_new + pop_comeback, self.problem.sense)[:pop_size])
