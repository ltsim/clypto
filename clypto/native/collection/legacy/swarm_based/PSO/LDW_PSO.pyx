#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.swarm_based.PSO._base cimport PSOAgent, ResetPSOPopulation


cdef class LDW_PSO(cy.Optimizer):
    """
    The original version of: Linearly Decreasing inertia Weight Particle Swarm Optimization (LDW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.8, 2.0], Weight max of bird, default = 0.9

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.LDW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w_min=0.4, w_max=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer." In 1998 IEEE international conference on
    evolutionary computation proceedings. IEEE world congress on computational intelligence (Cat. No. 98TH8360), pp. 69-73. IEEE, 1998.
    """

    cdef public double c1
    cdef public double c2
    cdef public double w_max
    cdef public double w_min

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        w_min: float = 0.4,
        w_max: float = 0.9,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
        """
        super().__init__(**kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=ResetPSOPopulation)
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.w_min = cy.validator(float, w_min, (0, 0.5), "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")
        self.parameters = ["epoch", "pop_size", "c1", "c2", "w_min", "w_max"]
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update weight after each move count  (weight down)
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        cdef PSOAgent agent
        for agent in self.population:
            agent.update_velocity(self.g_best.solution, w, self.c1, self.c2, self.generator,
                                  self.population.v_min, self.population.v_max)
            candidate = self.population.evaluate_solution(self.population.correct_solution(agent.move()))
            if cy.is_better(candidate, agent, self.problem.sense):
                agent.update_solution(candidate)
            agent.update_pbest(candidate, self.problem.sense)
