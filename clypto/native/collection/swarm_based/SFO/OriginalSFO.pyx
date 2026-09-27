#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalSFO(cy.Optimizer):
    """
    The original version of: SailFish Optimizer (SFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.01.001

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        + AP (float): coefficient for decreasing the value of Attack Power linearly from AP to 0
        + epsilon (float): should be 0.0001, 0.001

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import SFO    >>> import numpy as np
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
    >>> model = SFO.OriginalSFO(epoch=1000, pop_size=50, pp = 0.1, AP = 4.0, epsilon = 0.0001)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Shadravan, S., Naji, H.R. and Bardsiri, V.K., 2019. The Sailfish Optimizer: A novel nature-inspired metaheuristic
    algorithm for solving constrained engineering optimization problems. Engineering Applications of Artificial Intelligence, 80, pp.20-34.
    """

    cdef public double AP
    cdef public double epsilon
    cdef public double pp

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pp: float = 0.1,
            AP: float = 4.0,
            epsilon: float = 0.0001,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
            AP (float): coefficient for decreasing the value of Power Attack linearly from AP to 0
            epsilon (float): should be 0.0001, 0.001
        """
        super().__init__(parameters=["epoch", "pop_size", "pp", "AP", "epsilon"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pp = cy.validator(float, pp, (0, 1.0), "pp")
        self.AP = cy.validator(float, AP, (0, 100), "AP")
        self.epsilon = cy.validator(float, epsilon, (0, 0.1), "epsilon")
        self.s_size = int(self.population.size() / self.pp)

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)  # pop = sailfish
        self.s_pop = self.population.generate(self.s_size)
        self.s_gbest = cy.duplicate_agent(cy.sort_agents(self.s_pop, self.problem.sense)[0])  # s_pop = sardines

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        PD = 1 - pop_size / (pop_size + self.s_size)
        for idx in range(0, pop_size):
            lamda_i = 2 * self.generator.uniform() * PD - PD
            x = self.s_gbest.solution - lamda_i * (
                    self.generator.uniform()
                    * (self.population[idx].solution + self.s_gbest.solution)
                    / 2
                    - self.population[idx].solution
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
        ## Calculate AttackPower using Eq.(10)
        AP = self.AP * (1.0 - 2.0 * epoch * self.epsilon)
        if AP < 0.5:
            alpha = int(self.s_size * np.abs(AP))
            beta = int(self.problem.n_dims * np.abs(AP))
            ### Random self.generator.choice number of sardines which will be updated their position
            list1 = self.generator.choice(range(0, self.s_size), alpha)
            for idx in range(0, self.s_size):
                if idx in list1:
                    #### Random self.generator.choice number of dimensions in sardines updated, remove third loop by numpy vector computation
                    x = self.s_pop[idx].solution.copy()
                    list2 = self.generator.choice(
                        range(0, self.problem.n_dims), beta, replace=False
                    )
                    x[list2] = (
                            self.generator.uniform(0, 1, self.problem.n_dims)
                            * (self.s_gbest.solution - self.s_pop[idx].solution + AP)
                    )[list2]
                    x = cy.correct_solution(self.problem, x)
                    agent = self.population.create_agent(x)
                    if self.mode == "sequential":
                        agent.evaluate(self.problem)
                        self.s_pop[idx] = agent
        else:
            ### Update the position of all sardine using Eq.(9)
            for idx in range(0, self.s_size):
                x = self.generator.uniform() * (
                        self.g_best.solution - self.s_pop[idx].solution + AP
                )
                x = cy.correct_solution(self.problem, x)
                agent = self.population.create_agent(x)
                if self.mode == "sequential":
                    agent.evaluate(self.problem)
                    self.s_pop[idx] = agent
        ## Recalculate the fitness of all sardine
        self.s_pop = self.population.evaluate(self.s_pop, self.mode)
        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.population = self.population.sort()[:pop_size]
        self.s_pop = cy.sort_agents(self.s_pop, self.problem.sense)[:len(self.s_pop)]
        for idx, agent in enumerate(self.population.toarray()):
            for jdx in range(0, self.s_size):
                ### If there is a better position in sardine population.
                if cy.is_better(self.s_pop[jdx], agent, self.problem.sense):
                    self.population[idx] = cy.duplicate_agent(self.s_pop[jdx])
                    del self.s_pop[jdx]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        temp = self.s_size - len(self.s_pop)
        if temp == 1:
            self.s_pop = self.s_pop + [self.population.generate_agent()]
        else:
            self.s_pop = self.s_pop + self.population.generate(self.s_size - len(self.s_pop))
        self.s_gbest = cy.duplicate_agent(cy.sort_agents(self.s_pop, self.problem.sense)[0])
