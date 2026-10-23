#!/usr/bin/env python
# Created by "Thieu" at 10:08, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalHC(cy.Optimizer):
    """
    The original version of: Hill Climbing (HC)

    Notes:
        + The number of neighbour solutions are equal to user defined
        + The step size to calculate neighbour group is randomized
        + HC is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + neighbour_size (int): [2, 1000], fixed parameter, sensitive exploitation parameter, Default: 50

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import HC    >>> import numpy as np
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
    >>> model = HC.OriginalHC(epoch=1000, pop_size=50, neighbour_size = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mitchell, M., Holland, J. and Forrest, S., 1993. When will a genetic algorithm
    outperform hill climbing. Advances in neural information processing systems, 6.
    """

    cdef public int neighbour_size

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 2,
            neighbour_size: int = 50,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 2
            neighbour_size (int): fixed parameter, sensitive exploitation parameter, Default: 50
        """
        super().__init__(parameters=["epoch", "pop_size", "neighbour_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[2, 10000])
        self.neighbour_size = cy.validator(int, neighbour_size, [2, 1000], "neighbour_size")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        step_size = np.exp(-2 * epoch / self.epoch)
        pop_neighbours = []
        for idx in range(0, self.neighbour_size):
            x = (
                    self.g_best.solution
                    + self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up) * step_size
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_neighbours.append(agent)
            if self.mode == "sequential":
                pop_neighbours[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(pop_neighbours, self.mode))
