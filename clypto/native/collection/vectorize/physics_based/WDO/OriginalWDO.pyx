#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalWDO(VectorizeOptimizer):
    """
    The original version of: Wind Driven Optimization (WDO)

    Notes
        + pop is the set of "air parcel" - "position"
        + air parcel: is the set of gas atoms. Each atom represents a dimension in position and has its own velocity
        + pressure represented by fitness value
        + https://ieeexplore.ieee.org/abstract/document/6407788

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + RT (int): [2, 3, 4], RT coefficient, default = 3
        + g_c (float): [0.1, 0.5], gravitational constant, default = 0.2
        + alp (float): [0.3, 0.8], constants in the update equation, default=0.4
        + c_e (float): [0.1, 0.9], coriolis effect, default=0.4
        + max_v (float): [0.1, 0.9], maximum allowed speed, default=0.3

    Examples
    ~~~~~~~~

    >>> from clypto.native.collection.vectorize.physics_based import WDO    >>> import numpy as np
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
    >>> model = WDO.OriginalWDO(epoch=1000, pop_size=50, RT = 3, g_c = 0.2, alp = 0.4, c_e = 0.4, max_v = 0.3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Bayraktar, Z., Komurcu, M., Bossard, J.A. and Werner, D.H., 2013. The wind driven optimization
    technique and its application in electromagnetics. IEEE transactions on antennas and
    propagation, 61(5), pp.2745-2757.
    """

    cdef public object RT
    cdef public object g_c
    cdef public object alp
    cdef public object c_e
    cdef public object max_v
    cdef public object dyn_list_velocity

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        RT: int = 3,
        g_c: float = 0.2,
        alp: float = 0.4,
        c_e: float = 0.4,
        max_v: float = 0.3,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            RT (int): RT coefficient, default = 3
            g_c (float): gravitational constant, default = 0.2
            alp (float): constants in the update equation, default=0.4
            c_e (float): coriolis effect, default=0.4
            max_v (float): maximum allowed speed, default=0.3
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "RT", "g_c", "alp", "c_e", "max_v"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.RT = cy.validator(int, RT, [1, 4], "RT")
        self.g_c = cy.validator(float, g_c, (0, 1.0), "g_c")
        self.alp = cy.validator(float, alp, (0, 1.0), "alp")
        self.c_e = cy.validator(float, c_e, (0, 1.0), "c_e")
        self.max_v = cy.validator(float, max_v, (0, 1.0), "max_v")

    def _initialize_variables(self):
        self.dyn_list_velocity = self.max_v * self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up, (self.pop_size, self.problem.n_dims)
        )

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        rand_dim = rng.integers(0, d, size=n)
        V = self.dyn_list_velocity
        temp = V[np.arange(n), rand_dim][:, None] * np.ones((1, d))
        i1 = np.arange(1, n + 1)
        vel = (
                (1 - self.alp) * V
                - self.g_c * X
                + ((1 - 1.0 / i1) * self.RT)[:, None] * (g - X)
                + self.c_e * temp / i1[:, None]
        )
        vel = np.clip(vel, -self.max_v, self.max_v)
        V[:] = vel
        ops.step(self, X + vel)
