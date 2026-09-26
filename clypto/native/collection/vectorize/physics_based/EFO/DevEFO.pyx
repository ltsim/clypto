#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevEFO(LegacyNativeOptimizer):
    """
    The developed version: Electromagnetic Field Optimization (EFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EFO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = EFO.DevEFO(epoch=1000, pop_size=50, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r_rate: float = 0.3,
        ps_rate: float = 0.85,
        p_field: float = 0.1,
        n_field: float = 0.45,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "r_rate", "ps_rate", "p_field", "n_field"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.r_rate = cy.validator(float, r_rate, (0, 1.0), "r_rate")
        self.ps_rate = cy.validator(float, ps_rate, (0, 1.0), "ps_rate")
        self.p_field = cy.validator(float, p_field, (0, 1.0), "p_field")
        self.n_field = cy.validator(float, n_field, (0, 1.0), "n_field")
        self.phi = (1 + np.sqrt(5)) / 2

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        # random top, middle and bottom agents of the (sorted) population
        r1 = rng.integers(0, int(n * self.p_field), size=n)
        r2 = rng.integers(int(n * (1 - self.n_field)), n, size=n)
        r3 = rng.integers(int((n * self.p_field) + 1), int(n * (1 - self.n_field)), size=n)
        pos = np.where(
            (rng.random(n) < self.ps_rate)[:, None],
            X[r1] + self.phi * rng.random((n, 1)) * (g - X[r3]) + rng.random((n, 1)) * (g - X[r2]),
            lb + rng.random((n, d)) * (ub - lb),
        )
        # random re-initialization of one coordinate
        mutate = rng.random(n) < self.r_rate
        ri = rng.integers(0, d, size=n)
        col = rng.integers(0, d, size=n)
        val = lb[ri] + rng.random(n) * (ub[ri] - lb[ri])
        rows = np.flatnonzero(mutate)
        pos[rows, col[rows]] = val[rows]
        ops.step(self, pos)
