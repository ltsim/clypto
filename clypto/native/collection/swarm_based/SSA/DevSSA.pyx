#!/usr/bin/env python
# Created by "Thieu" at 17:22, 29/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevSSA(cy.Optimizer):
    """
    The developed version: Sparrow Search Algorithm (SSA)

    Notes:
        + First, the population is sorted to find g-best and g-worst
        + In Eq. 4, the self.generator.normal() gaussian distribution is used instead of A+ and L

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
        + PD (float): number of producers (percentage), default = 0.2
        + SD (float): number of sparrows who perceive the danger, default = 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import SSA    >>> import numpy as np
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
    >>> model = SSA.DevSSA(epoch=1000, pop_size=50, ST = 0.8, PD = 0.2, SD = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Xue, J. and Shen, B., 2020. A novel swarm intelligence optimization approach:
    sparrow search algorithm. Systems Science & Control Engineering, 8(1), pp.22-34.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            ST: float = 0.8,
            PD: float = 0.2,
            SD: float = 0.1,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
            PD (float): number of producers (percentage), default = 0.2
            SD (float): number of sparrows who perceive the danger, default = 0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "ST", "PD", "SD"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=cy.Population)
        self.ST = cy.validator(float, ST, (0, 1.0), "ST")
        self.PD = cy.validator(float, PD, (0, 1.0), "PD")
        self.SD = cy.validator(float, SD, (0, 1.0), "SD")
        self.n1 = int(self.PD * self.population.size())
        self.n2 = int(self.SD * self.population.size())

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        r2 = self.generator.uniform()  # R2 in [0, 1], the alarm value, random value
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            # Using equation (3) update the sparrow’s location;
            if idx < self.n1:
                if r2 < self.ST:
                    des = epoch / (self.generator.uniform() * self.epoch + self.EPSILON)
                    if des > 5:
                        des = self.generator.normal()
                    x_new = self.population[idx].solution * np.exp(des)
                else:
                    x_new = self.population[idx].solution + self.generator.normal() * np.ones(
                        self.problem.n_dims
                    )
            else:
                # Using equation (4) update the sparrow’s location;
                ranked = self.population.sort()
                (g_best,) = [cy.duplicate_agent(agent) for agent in ranked[:1]]
                (g_worst,) = [cy.duplicate_agent(agent) for agent in ranked[::-1][:1]]
                if idx > int(pop_size / 2):
                    x_new = self.generator.normal() * np.exp(
                        (g_worst.solution - self.population[idx].solution) / (idx + 1) ** 2
                    )
                else:
                    x_new = (
                            g_best.solution
                            + np.abs(self.population[idx].solution - g_best.solution)
                            * self.generator.normal()
                    )
            x = cy.reset_solution(self.problem, self.generator, x_new)
            agent = self.population.create_agent(x)
            n_population.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
        self.population = self.population.sort()
        best = [cy.duplicate_agent(agent) for agent in self.population[:1]]
        worst = [cy.duplicate_agent(agent) for agent in self.population[::-1][:1]]
        g_best, g_worst = best[0], worst[0]
        pop2 = [cy.duplicate_agent(agent) for agent in self.population[self.n2:]]
        child = []
        for idx in range(0, len(pop2)):
            #  Using equation (5) update the sparrow’s location;
            if cy.is_better(self.population[idx], g_best, self.problem.sense):
                x_new = pop2[idx].solution + self.generator.uniform(-1, 1) * (
                        np.abs(pop2[idx].solution - g_worst.solution)
                        / (pop2[idx].fitness - g_worst.fitness + self.EPSILON)
                )
            else:
                x_new = g_best.solution + self.generator.normal() * np.abs(
                    pop2[idx].solution - g_best.solution
                )
            x = cy.reset_solution(self.problem, self.generator, x_new)
            agent = self.population.create_agent(x)
            child.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                pop2[idx] = cy.get_better_agent(pop2[idx], agent, self.problem.sense)
        if self.mode != "sequential":
            child = self.population.evaluate(child, self.mode)
            pop2 = cy.greedy_agents(pop2, child, self.problem.sense)
        self.population = self.population.spawn(self.population[: self.n2] + pop2)
