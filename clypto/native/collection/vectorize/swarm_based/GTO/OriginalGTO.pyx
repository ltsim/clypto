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


cdef class OriginalGTO(LegacyNativeOptimizer):
    """
    The original version of: Giant Trevally Optimizer (GTO)

    Notes:
        1. This version is implemented exactly as described in the paper.
        2. https://www.mathworks.com/matlabcentral/fileexchange/121358-giant-trevally-optimizer-gto
        3. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9955508

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + A (float): a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
        + H (float): initial value for specifies the jumping slope function, default=2.0

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
    >>> model = GTO.OriginalGTO(epoch=1000, pop_size=50, A=0.4, H=2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sadeeq, H. T., & Abdulazeez, A. M. (2022). Giant Trevally Optimizer (GTO): A Novel Metaheuristic
    Algorithm for Global Optimization and Challenging Engineering Problems. IEEE Access, 10, 121615-121640.
    """

    cdef public object A
    cdef public object H

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        A: float = 0.4,
        H: float = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            A (float): a position-change-controlling parameter with a range from 0.3 to 0.4, default=0.4
            H (float): initial value for specifies the jumping slope function, default=2.0
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "A", "H"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.A = cy.validator(float, A, [-10.0, 10.0], "A")
        self.H = cy.validator(float, H, [1.0, 10.0], "H")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        lb, ub = self.problem.lb, self.problem.ub
        # Step 1: extensive search, Eq.(4)
        g = np.array(self.g_best_x())
        levy = self.get_levy_flight_step(beta=1.5, multiplier=0.01, size=(n, d), case=-1)
        ops.step(self, g * rng.random((n, 1)) + ((ub - lb) * rng.random((n, 1)) + lb) * levy)
        # Step 2: choosing area, Eq. 7 (around the current best)
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        r3 = rng.random((n, 1))
        ops.step(self, g * self.A * r3 + np.mean(np.ascontiguousarray(X), axis=0) - X * r3)
        # Step 3: attacking, Eqs. 10-15
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        H = rng.random() * self.H * (1 - epoch_c / self.epoch)
        dist = np.sum(np.abs(g - X), axis=1, keepdims=True)
        theta2 = 360 * rng.random((n, 1))
        theta1 = (1.33 / 1.00029) * np.sin(np.radians(theta2))
        VD = np.sin(np.radians(theta1)) * dist
        ops.step(self, X * np.sin(np.radians(theta2)) * np.asarray(pop.F)[:, None] + VD + H)
