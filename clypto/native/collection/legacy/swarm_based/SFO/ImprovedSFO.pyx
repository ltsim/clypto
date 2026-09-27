cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 14:51, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class ImprovedSFO(cy.Optimizer):
    """
    The original version: Improved Sailfish Optimizer (I-SFO)

    Notes:
        + Energy equation is reformed
        + AP (A) and epsilon parameters are removed
        + Opposition-based learning technique is used

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import SFO    >>> import numpy as np
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
    >>> model = SFO.ImprovedSFO(epoch=1000, pop_size=50, pp = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double pp

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, pp: float = 0.1, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, SailFish pop size
            pp (float): the rate between SailFish and Sardines (N_sf = N_s * pp) = 0.25, 0.2, 0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "pp"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pp = cy.validator(float, pp, (0, 1.0), "pp")
        self.s_size = int(self.population.size() / self.pp)

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.s_pop = self.population.generate(self.s_size)
        self.s_gbest = cy.sort_agents(self.s_pop, self.problem.sense)[0].copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Calculate lamda_i using Eq.(7)
        ## Update the position of sailfish using Eq.(6)
        pop_new = []
        for idx in range(0, pop_size):
            PD = 1 - len(self.population) / (len(self.population) + len(self.s_pop))
            lamda_i = 2 * self.generator.uniform() * PD - PD
            pos_new = self.s_gbest.solution - lamda_i * (
                    self.generator.uniform()
                    * (self.g_best.solution + self.s_gbest.solution)
                    / 2
                    - self.population[idx].solution
            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        ## ## Calculate AttackPower using my Eq.thieu
        #### This is our proposed, simple but effective, no need A and epsilon parameters
        AP = 1 - epoch * 1.0 / self.epoch
        if AP < 0.5:
            for idx in range(0, len(self.s_pop)):
                temp = (self.g_best.solution + AP) / 2
                pos_new = (
                        self.problem.bounds.low
                        + self.problem.bounds.up
                        - temp
                        + self.generator.uniform() * (temp - self.s_pop[idx].solution)
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.s_pop[idx] = agent
        else:
            ### Update the position of all sardine using Eq.(9)
            for idx in range(0, len(self.s_pop)):
                pos_new = self.generator.uniform() * (
                        self.g_best.solution - self.s_pop[idx].solution + AP
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.s_pop[idx] = agent
        ## Recalculate the fitness of all sardine
        self.s_pop = self.population.evaluate(self.s_pop, self.mode)
        ## Sort the population of sailfish and sardine (for reducing computational cost)
        self.population = self.population.sort()[:pop_size]
        self.s_pop = cy.sort_agents(self.s_pop, self.problem.sense)[:len(self.s_pop)]
        for idx in range(0, pop_size):
            for jdx in range(0, len(self.s_pop)):
                ### If there is a better position in sardine population.
                if cy.is_better(self.s_pop[jdx], self.population[idx], self.problem.sense):
                    self.population[idx] = self.s_pop[jdx].copy()
                    del self.s_pop[jdx]
                break  #### This simple keyword helped reducing ton of comparing operation.
                #### Especially when sardine pop size >> sailfish pop size
        self.s_pop = self.s_pop + self.population.generate(self.s_size - len(self.s_pop))
        self.s_gbest = cy.sort_agents(self.s_pop, self.problem.sense)[0].copy()
