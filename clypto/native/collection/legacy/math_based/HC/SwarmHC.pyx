#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class SwarmHC(cy.Optimizer):
    """
    The developed version: Swarm-based Hill Climbing (S-HC)

    Notes
    ~~~~~
    + Based on swarm-of people are trying to climb on the mountain idea
    + The number of neighbour solutions are equal to population size
    + The step size to calculate neighbour is randomized and based on rank of solution.
        + The guys near on top of mountain will move slower than the guys on bottom of mountain.
        + Imagination: exploration when far from global best, and exploitation when near global best
    + Who on top of mountain first will be the winner. (global optimal)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, pop_size/2], fixed parameter, sensitive exploitation parameter, Default: 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.math_based import HC    >>> import numpy as np
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
    >>> model = HC.SwarmHC(epoch=1000, pop_size=50, neighbour_size = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public int neighbour_size

    def __init__(self, epoch=10000, pop_size=100, neighbour_size=10, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 10
        """
        super().__init__(parameters=["epoch", "pop_size", "neighbour_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.neighbour_size = cy.validator(int, neighbour_size, [2, int(self.population.size() / 2)], "neighbour_size")

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ranks = np.array(list(range(1, pop_size + 1)))
        ranks = ranks / np.sum(ranks)
        step_size = np.mean(self.problem.bounds.up - self.problem.bounds.low) * np.exp(
            -2 * epoch / self.epoch
        )
        ss = step_size * ranks
        pop = []
        for idx in range(0, pop_size):
            pop_neighbours = []
            for jdx in range(0, self.neighbour_size):
                pos_new = (
                        self.population[idx].solution
                        + self.generator.normal(0, 1, self.problem.n_dims) * ss[idx]
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                pop_neighbours.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    pop_neighbours[-1].evaluate(self.problem)
            pop_neighbours = self.population.evaluate(pop_neighbours, self.mode)
            best_local = cy.sort_agents(pop_neighbours, self.problem.sense)[0].copy()
            pop.append(best_local)
            if self.mode not in self.AVAILABLE_MODES:
                self.population[idx] = cy.get_better_agent(best_local, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            self.population = self.population.greedy(pop)
