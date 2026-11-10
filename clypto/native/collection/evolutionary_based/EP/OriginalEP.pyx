#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy
from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory





cdef class OriginalEPAgents(AgentFactory):
    """The agent factory of ``OriginalEP`` (moved off the optimizer)."""
    cdef public object distance

    def __init__(self, distance=None):
        self.distance = distance

    cdef Agent create(self, object solution=None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance, self.problem.n_dims)
        times_win = 0

        agent = cy.Agent(solution)
        agent.strategy = strategy
        agent.win = times_win
        return agent

    cdef Agent generate(self, object solution=None):
        agent = self.create(solution)
        agent.evaluate(self.problem)
        return agent


cdef class OriginalEP(cy.Optimizer):
    """
    The original version of: Evolutionary Programming (EP)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

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
    >>> model = EP.OriginalEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Yao, X., Liu, Y. and Lin, G., 1999. Evolutionary programming made faster.
    IEEE Transactions on Evolutionary computation, 3(2), pp.82-102.
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
        super().__init__(parameters=["epoch", "pop_size", "bout_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])
        self.bout_size = cy.validator(float, bout_size, (0, 1.0), "bout_size")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.n_bout_size = int(self.bout_size * pop_size)
        self.distance = 0.05 * (self.problem.bounds.up - self.problem.bounds.low)
        self.population.factory = OriginalEPAgents(self.distance)

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
            agent = self.population.create_agent(x)
            s_old = (
                self.population[idx].strategy
                + self.generator.normal(0, 1.0, self.problem.n_dims)
                * np.abs(self.population[idx].strategy) ** 0.5
            )
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
        pop = sorted(pop, key=lambda agent: agent.win, reverse=True)
        self.population = self.population.spawn(pop[: pop_size])


