cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalAEO(cy.Optimizer):
    """
    The original version of: Artificial Ecosystem-based Optimization (AEO)

    Links:
        1. https://doi.org/10.1007/s00521-019-04452-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/72685-artificial-ecosystem-based-optimization-aeo

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.OriginalAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2020. Artificial ecosystem-based optimization: a novel
    nature-inspired meta-heuristic algorithm. Neural Computing and Applications, 32(13), pp.9383-9425.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * self.generator.uniform()
        x1 = (1 - a) * self.population[-1].solution + a * self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up
        )
        x = cy.correct_solution(self.problem, x1)
        agent = self.population.generate_agent(x)
        self.population[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            jdx = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                x_t1 = self.population[idx].solution + c * (
                        self.population[idx].solution - self.population[0].solution
                )  # Eq. 6
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                x_t1 = self.population[idx].solution + c * (
                        self.population[idx].solution - self.population[jdx].solution
                )  # Eq. 7
            ### Omnivore
            else:
                r2 = self.generator.uniform()
                x_t1 = self.population[idx].solution + c * (
                        r2 * (self.population[idx].solution - self.population[0].solution)
                        + (1 - r2) * (self.population[idx].solution - self.population[jdx].solution)
                )
            x = cy.correct_solution(self.problem, x_t1)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population[:-1] = cy.greedy_agents(self.population[:-1], pop_new, self.problem.sense)
        ## find current best used in decomposition
        best = cy.duplicate_agent(self.population.sort()[0])
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx, agent in enumerate(self.population.toarray()):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            x_t1 = best.solution + d * (e * best.solution - h * agent.solution)
            x = cy.correct_solution(self.problem, x_t1)
            child = self.population.create_agent(x)
            pop_child.append(child)
        self.population = self.population.greedy(self.population.evaluate(pop_child, self.mode), self.mode)
