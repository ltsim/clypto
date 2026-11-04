#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.swarm_based.PSO._base cimport PSOAgent, PSOPopulation


cdef class OriginalPSO(cy.Optimizer):
    """
    The original version of: Particle Swarm Optimization (PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w (float): (0., 1.0), Weight min of bird, default = 0.4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import PSO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = PSO.OriginalPSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Kennedy, J. and Eberhart, R., 1995, November. Particle swarm optimization. In Proceedings of
    ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
    """

    cdef public double c1
    cdef public double c2
    cdef public double w

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        w: float = 0.4,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
        """
        super().__init__(parameters=["epoch", "pop_size", "c1", "c2", "w"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=PSOPopulation)
        self.c1 = cy.validator(float, c1, (0.0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0.0, 5.0), "c2")
        self.w = cy.validator(float, w, (0.0, 1.0), "w")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm.

        Args:
            epoch (int): The current iteration
        """
        cdef PSOAgent agent
        # sequential on purpose: g_best is one of the particles, and update_solution moves it in place,
        # so the particles after it already follow the new g_best (a batch step would change results)
        for agent in self.population.toarray():
            # 1. The particle updates its velocity and proposes a position inside the bounds
            agent.update_velocity(self.g_best.solution, self.w, self.c1, self.c2, self.generator)
            # 2. Evaluate that position as a candidate
            candidate = self.population.evaluate_solution(cy.reset_solution(self.problem, self.generator, agent.move()))
            # 3. Greedy: move only if better, then refresh the personal best
            if cy.is_better(candidate, agent, self.problem.sense):
                agent.update_solution(candidate)
            agent.update_pbest(candidate, self.problem.sense)
