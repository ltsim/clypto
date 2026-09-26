#!/usr/bin/env python
# Created by "Thieu" at 21:45, 26/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalAVOA(VectorizeOptimizer):
    """
    The original version of: African Vultures Optimization Algorithm (AVOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0360835221003120
        2. https://www.mathworks.com/matlabcentral/fileexchange/94820-african-vultures-optimization-algorithm

    Notes (parameters):
        + p1 (float): probability of status transition, default 0.6
        + p2 (float): probability of status transition, default 0.4
        + p3 (float): probability of status transition, default 0.6
        + alpha (float): probability of 1st best, default = 0.8
        + gama (float): a factor in the paper (not much affect to algorithm), default = 2.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import AVOA    >>> import numpy as np
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
    >>> model = AVOA.OriginalAVOA(epoch=1000, pop_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new
    nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.
    """

    cdef public object p1
    cdef public object p2
    cdef public object p3
    cdef public object alpha
    cdef public object gama

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p1: float = 0.6,
        p2: float = 0.4,
        p3: float = 0.6,
        alpha: float = 0.8,
        gama: float = 2.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "p1", "p2", "p3", "alpha", "gama"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p1 = cy.validator(float, p1, (0, 1), "p1")
        self.p2 = cy.validator(float, p2, (0, 1), "p2")
        self.p3 = cy.validator(float, p3, (0, 1), "p3")
        self.alpha = cy.validator(float, alpha, (0, 1), "alpha")
        self.gama = cy.validator(float, gama, (0, 5.0), "gama")

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        a = rng.uniform(-2, 2) * ((np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama) + np.cos((np.pi / 2) * (epoch / self.epoch)) - 1)
        ppp = (2 * rng.random() + 1) * (1 - epoch / self.epoch) + a
        order = self.sorted_order(pop)
        best = np.array(X[order[:2]])
        F = ppp * (2 * rng.random((n, 1)) - 1)
        rand_pos = best[rng.choice([0, 1], p=[self.alpha, 1 - self.alpha], size=n)]
        R = rng.random((n, 9, 1))
        # exploration (|F| >= 1)
        p_a = rand_pos - np.abs((2 * R[:, 0]) * rand_pos - X) * F
        p_b = rand_pos - F + R[:, 1] * ((ub - lb) * R[:, 2] + lb)
        explore = np.where(R[:, 3] < self.p1, p_a, p_b)
        # exploitation, phase 1 (|F| < 0.5)
        A = best[0] - ((best[0] * X) / (best[0] - X ** 2 + self.EPSILON)) * F
        B = best[1] - ((best[1] * X) / (best[1] - X ** 2 + self.EPSILON)) * F
        levy = self._get_levy_flight_step(beta=1.5, multiplier=1.0, size=(n, X.shape[1]), case=-1)
        phase1 = np.where(R[:, 4] < self.p2, (A + B) / 2, rand_pos - np.abs(rand_pos - X) * F * levy)
        # exploitation, phase 2 (0.5 <= |F| < 1)
        s = rand_pos * (R[:, 6] * X / (2 * np.pi))
        phase2 = np.where(R[:, 5] < self.p3,
                          np.abs((2 * R[:, 7]) * rand_pos - X) * (F + R[:, 8]) - (rand_pos - X),
                          rand_pos - (s * np.cos(X) + s * np.sin(X)))
        absF = np.abs(F)
        ops.replace(self, np.where(absF >= 1, explore, np.where(absF < 0.5, phase1, phase2)))
