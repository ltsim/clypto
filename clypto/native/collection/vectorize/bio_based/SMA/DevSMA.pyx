#!/usr/bin/env python
# Created by "Thieu" at 20:22, 12/06/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevSMA(LegacyNativeOptimizer):
    """
    The developed version: Slime Mould Algorithm (SMA)

    Notes:
        + Selected 2 unique and random solution to create new solution (not to create variable)
        + Check bound and compare old position with new position to get the best one

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_t (float): (0, 1.0) -> better [0.01, 0.1], probability threshold (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SMA    >>> import numpy as np
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
    >>> model = SMA.DevSMA(epoch=1000, pop_size=50, p_t = 0.03)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p_t: float = 0.03,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_t (float): probability threshold (z in the paper), default = 0.03
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "p_t"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p_t = cy.validator(float, p_t, (0, 1.0), "p_t")

    cdef void initialize_variables(self):
        self.weights = np.zeros((self.pop_size, self.problem.n_dims))

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        ss = gb_fit - pop.F[-1] + self.EPSILON
        lb, ub = self.problem.lb, self.problem.ub
        # weights (Eq. 2.5): the better half is amplified, the worse half is damped
        sign = np.where(np.arange(n) <= int(self.pop_size / 2), 1.0, -1.0)[:, None]
        self.weights = 1 + sign * rng.uniform(0, 1, (n, d)) * np.log10((gb_fit - np.asarray(pop.F)) / ss + 1)[:, None]
        a = np.arctanh(1 - epoch / self.epoch)  # Eq.(2.4)
        b = 1 - epoch / self.epoch
        p = np.tanh(np.abs(np.asarray(pop.F) - gb_fit))[:, None]  # Eq.(2.2)
        vb = rng.uniform(-a, a, (n, d))  # Eq.(2.3)
        vc = rng.uniform(-b, b, (n, d))
        ia, ib = ops.two_others(self, n, 1)
        pos_1 = g + vb * (self.weights * X[ia[:, 0]] - X[ib[:, 0]])
        pos_2 = vc * X
        pos = np.where(rng.random((n, d)) < p, pos_1, pos_2)
        pos = np.where((rng.random(n) < self.p_t)[:, None], lb + rng.random((n, d)) * (ub - lb), pos)  # Eq.(2.7)
        ops.step(self, pos)
