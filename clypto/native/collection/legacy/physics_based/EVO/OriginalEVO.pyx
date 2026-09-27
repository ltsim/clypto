#!/usr/bin/env python
# Created by "Thieu" at 18:09, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalEVO(cy.Optimizer):
    """
    The original version of: Energy Valley Optimizer (EVO)

    Links:
        1. https://www.nature.com/articles/s41598-022-27344-y
        2. https://www.mathworks.com/matlabcentral/fileexchange/123130-energy-valley-optimizer-a-novel-metaheuristic-algorithm

    Notes:
        1. The algorithm is straightforward and does not require any specialized knowledge or techniques.
        2. The algorithm may not perform optimally due to slow convergence and no good operations, which could be improved by implementing better strategies and operations.
        3. The problem is that it is stuck at a local optimal around 1/2 of the max generations because fitness distance is being used as a factor in the equations.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import EVO    >>> import numpy as np
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
    >>> model = EVO.OriginalEVO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Azizi, M., Aickelin, U., A. Khorshidi, H., & Baghalzadeh Shishehgarkhaneh, M. (2023). Energy valley optimizer: a novel
    metaheuristic algorithm for global and engineering optimization. Scientific Reports, 13(1), 226.
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

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx in range(0, pop_size):
            pos_list = np.array([agent.solution for agent in self.population])
            fit_list = np.array([agent.fitness for agent in self.population])
            dis = np.sqrt(np.sum((self.population[idx].solution - pos_list) ** 2, axis=1))
            idx_dis_sort = np.argsort(dis)
            CnPtIdx = self.generator.choice(list(set(range(2, pop_size)) - {idx}))
            x_team = pos_list[idx_dis_sort[1:CnPtIdx], :]
            x_avg_team = np.mean(x_team, axis=0)
            x_avg_pop = np.mean(pos_list, axis=0)
            eb = np.mean(fit_list)
            sl = (fit_list[idx] - self.g_best.fitness) / (
                    self.g_worst.fitness - self.g_best.fitness + self.EPSILON
            )

            pos_new1 = self.population[idx].solution.copy()
            pos_new2 = self.population[idx].solution.copy()
            if cy.better_fitness(eb, self.population[idx].fitness, self.problem.sense):
                if self.generator.random() > sl:
                    a1_idx = self.generator.integers(self.problem.n_dims)
                    a2_idx = self.generator.integers(
                        0, self.problem.n_dims, size=a1_idx
                    )
                    pos_new1[a2_idx] = self.g_best.solution[a2_idx]
                    g1_idx = self.generator.integers(self.problem.n_dims)
                    g2_idx = self.generator.integers(
                        0, self.problem.n_dims, size=g1_idx
                    )
                    pos_new2[g2_idx] = x_avg_team[g2_idx]
                else:
                    ir = self.generator.uniform(0, 1, 2)
                    jr = self.generator.uniform(0, 1, self.problem.n_dims)
                    pos_new1 += (
                            jr * (ir[0] * self.g_best.solution - ir[1] * x_avg_pop) / sl
                    )
                    ir = self.generator.uniform(0, 1, 2)
                    jr = self.generator.uniform(0, 1, self.problem.n_dims)
                    pos_new2 += jr * (ir[0] * self.g_best.solution - ir[1] * x_avg_team)
                pos_new1 = self.population.correct_solution(pos_new1)
                pos_new2 = self.population.correct_solution(pos_new2)
                agent1 = self.population.create_agent(pos_new1)
                agent2 = self.population.create_agent(pos_new2)
                pop_new.append(agent1)
                pop_new.append(agent2)
            else:
                pos_new = (
                        pos_new1
                        + self.generator.random()
                        * sl
                        * self.generator.uniform(
                    self.problem.bounds.low, self.problem.bounds.up, self.problem.n_dims
                )
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                pop_new.append(agent)
        if self.mode not in self.AVAILABLE_MODES:
            for idx in range(0, len(pop_new)):
                pop_new[idx].evaluate(self.problem)
        pop_new = self.population.evaluate(pop_new, self.mode)
        self.population = cy.sort_agents(self.population + pop_new, self.problem.sense)[:pop_size]
