#!/usr/bin/env python
# Created by "Thieu" at 15:37, 19/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalHGSAgent(cy.Agent):
    cdef public object hunger

    def __init__(self, solution=None, objectives=None, weights=None, hunger=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.hunger = hunger

    cdef cy.Agent clone(self):
        cdef OriginalHGSAgent new = <OriginalHGSAgent>cy.Agent.clone(self)
        new.hunger = self.hunger
        return new


cdef class OriginalHGSPopulation(cy.Population):
    """Agents of :class:`OriginalHGS`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        hunger = 1.0
        return OriginalHGSAgent(solution=solution, hunger=hunger)


cdef class OriginalHGS(cy.Optimizer):
    """
    The original version of: Hunger Games Search (HGS)

    Links:
        https://aliasgharheidari.com/HGS.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + PUP (float): [0.01, 0.2], The probability of updating position (L in the paper), default = 0.08
        + LH (float): [1000, 20000], Largest hunger / threshold, default = 10000

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import HGS    >>> import numpy as np
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
    >>> model = HGS.OriginalHGS(epoch=1000, pop_size=50, PUP = 0.08, LH = 10000)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, Y., Chen, H., Heidari, A.A. and Gandomi, A.H., 2021. Hunger games search: Visions, conception, implementation,
    deep analysis, perspectives, and towards performance shifts. Expert Systems with Applications, 177, p.114864.
    """

    cdef public double LH
    cdef public double PUP

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        PUP: float = 0.08,
        LH: float = 10000,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            PUP (float): The probability of updating position (L in the paper), default = 0.08
            LH (float): Largest hunger / threshold, default = 10000
        """
        super().__init__(parameters=["epoch", "pop_size", "PUP", "LH"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalHGSPopulation)
        self.PUP = cy.validator(float, PUP, (0, 1.0), "PUP")
        self.LH = cy.validator(float, LH, [1, 20000], "LH")

    def sech__(self, x):
        if np.abs(x) > 50:
            return 0.5
        return 2 / (np.exp(x) + np.exp(-x))

    def update_hunger_value__(self, pop=None, g_best=None, g_worst=None):
        pop_size = self.population.size()
        # min_index = pop.index(min(pop, key=lambda x: x.fitness))
        # Eq (2.8) and (2.9)
        for idx in range(0, pop_size):
            r = self.generator.random()
            # space: since we pass lower bound and upper bound as list. Better take the np.mean of them.
            space = np.mean(self.problem.bounds.up - self.problem.bounds.low)
            H = (
                (pop[idx].fitness - g_best.fitness)
                / (g_worst.fitness - g_best.fitness + self.EPSILON)
                * r
                * 2
                * space
            )
            if H < self.LH:
                H = self.LH * (1 + r)
            pop[idx].hunger += H

            if g_best.fitness == pop[idx].fitness:
                pop[idx].hunger = 0
        return pop

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Eq. (2.2)
        ### Find the current best and current worst
        ranked = self.population.sort()
        (g_best,) = [cy.duplicate_agent(agent) for agent in ranked[:1]]
        (g_worst,) = [cy.duplicate_agent(agent) for agent in ranked[::-1][:1]]
        pop = self.update_hunger_value__(self.population, g_best, g_worst)

        ## Eq. (2.4)
        shrink = 2 * (1 - epoch / self.epoch)
        total_hunger = np.sum([pop[idx].hunger for idx in range(0, pop_size)])

        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            child = cy.duplicate_agent(agent)
            #### Variation control
            E = self.sech__(agent.fitness - g_best.fitness)

            # R is a ranging controller added to limit the range of activity, in which the range of R is gradually reduced to 0
            R = 2 * shrink * self.generator.random() - shrink  # Eq. (2.3)

            ## Calculate the hungry weight of each position
            if self.generator.random() < self.PUP:
                W1 = (
                    agent.hunger
                    * pop_size
                    / (total_hunger + self.EPSILON)
                    * self.generator.random()
                )
            else:
                W1 = 1
            W2 = (
                (1 - np.exp(-np.abs(agent.hunger - total_hunger)))
                * self.generator.random()
                * 2
            )

            ### Udpate position of individual Eq. (2.1)
            r1 = self.generator.random()
            r2 = self.generator.random()
            if r1 < self.PUP:
                x = agent.solution * (1 + self.generator.normal(0, 1))
            else:
                if r2 > E:
                    x = W1 * g_best.solution + R * W2 * np.abs(
                        g_best.solution - agent.solution
                    )
                else:
                    x = W1 * g_best.solution - R * W2 * np.abs(
                        g_best.solution - agent.solution
                    )
            x = cy.correct_solution(self.problem, x)
            child.solution = x
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], child, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
