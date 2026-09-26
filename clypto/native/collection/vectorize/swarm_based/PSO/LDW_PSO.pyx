#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.native.collection.vectorize.swarm_based.PSO.OriginalPSO cimport OriginalPSO


cdef class LDW_PSO(OriginalPSO):
    """
    The original version of: Linearly Decreasing inertia Weight Particle Swarm Optimization (LDW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.8, 2.0], Weight max of bird, default = 0.9

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import PSO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = PSO.LDW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w_min=0.4, w_max=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shi, Yuhui, and Russell Eberhart. "A modified particle swarm optimizer." In 1998 IEEE international conference on
    evolutionary computation proceedings. IEEE world congress on computational intelligence (Cat. No. 98TH8360), pp. 69-73. IEEE, 1998.
    """

    cdef public double w_min
    cdef public double w_max

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        w_min: float = 0.4,
        w_max: float = 0.9,
        *,
        name: str | None = None,
        mode: str | None = None,
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
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c1", "c2", "w_min", "w_max"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.w_min = cy.validator(float, w_min, (0, 0.5), "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")

    cdef double weight(self, int epoch):
        # Update weight after each move count  (weight down)
        return (self.epoch - epoch) / <double>self.epoch * (self.w_max - self.w_min) + self.w_min

    cdef bint clips_velocity(self):
        return True
