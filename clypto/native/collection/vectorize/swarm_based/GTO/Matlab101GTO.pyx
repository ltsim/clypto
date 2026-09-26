#!/usr/bin/env python
# Created by "Thieu" at 21:58, 16/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class Matlab101GTO(LegacyNativeOptimizer):
    """
    The conversion of Matlab code (version 1.0.1 - 29/11/2022) to Python code of: Giant Trevally Optimizer (GTO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        2. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Notes:
        1. This algorithm costs a huge amount of computational resources in each epoch.
        Therefore, be careful when using the maximum number of generations as a stopping condition.
        2. Other algorithms update around K*pop_size times in each epoch, this algorithm updates around 2*pop_size^2 + pop_size times
        3. This version is used by the authors to compared with other algorithms in their paper.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import GTO    >>> import numpy as np
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
    >>> model = GTO.Matlab101GTO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d, m = pop.n * (pop.n - 1)
        cdef object rng = self.generator
        lb, ub = self.problem.lb, self.problem.ub
        parent = np.repeat(np.arange(n), n - 1)  # n - 1 candidates per agent
        # Step 1: extensive search, Eq.(4): every agent keeps the best of its candidates and itself
        g = np.array(self.g_best_x())
        levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=(m, d), case=-1)
        cand = pop.take(parent)
        cand.X[:] = self.correct_solution(g * rng.random((m, 1)) + ((ub - lb) * rng.random((m, 1)) + lb) * levy)
        self.evaluate(cand, 0, m)
        ops.scatter(self, cand, parent)
        # Step 2: choosing area, Eq. 7
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        r3 = rng.random((n, 1))
        ops.step(self, g * 0.4 * r3 + np.mean(np.ascontiguousarray(X), axis=0) - X * r3)
        # Step 3: attacking: every agent keeps the best of its candidates and itself
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        H = rng.random() * 2.0 * (1.0 - epoch_c / self.epoch)
        dist = np.sum(np.abs(g - X), axis=1)[parent][:, None]
        theta2 = 360 * rng.random((m, 1))
        theta1 = 1.3296 * np.sin(np.radians(theta2))
        VD = np.sin(np.radians(theta1)) * dist
        cand = pop.take(parent)
        cand.X[:] = self.correct_solution(X[parent] * np.sin(np.radians(theta2)) * np.asarray(pop.F)[parent][:, None] + VD + H)
        self.evaluate(cand, 0, m)
        ops.scatter(self, cand, parent)
