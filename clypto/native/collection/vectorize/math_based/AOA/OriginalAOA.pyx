#!/usr/bin/env python
# Created by "Thieu" at 09:56, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalAOA(VectorizeOptimizer):
    """
    The original version of: Arithmetic Optimization Algorithm (AOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2020.113609

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (int): [3, 8], fixed parameter, sensitive exploitation parameter, Default: 5,
        + miu (float): [0.3, 1.0], fixed parameter , control parameter to adjust the search process, Default: 0.5,
        + moa_min (float): [0.1, 0.4], range min of Math Optimizer Accelerated, Default: 0.2,
        + moa_max (float): [0.5, 1.0], range max of Math Optimizer Accelerated, Default: 0.9,

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.math_based import AOA    >>> import numpy as np
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
    >>> model = AOA.OriginalAOA(epoch=1000, pop_size=50, alpha = 5, miu = 0.5, moa_min = 0.2, moa_max = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M. and Gandomi, A.H., 2021. The arithmetic
    optimization algorithm. Computer methods in applied mechanics and engineering, 376, p.113609.
    """

    cdef public int alpha
    cdef public double miu
    cdef public double moa_min
    cdef public double moa_max

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 5,
        miu: float = 0.5,
        moa_min: float = 0.2,
        moa_max: float = 0.9,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): fixed parameter, sensitive exploitation parameter, Default: 5,
            miu (float): fixed parameter, control parameter to adjust the search process, Default: 0.5,
            moa_min (float): range min of Math Optimizer Accelerated, Default: 0.2,
            moa_max (float): range max of Math Optimizer Accelerated, Default: 0.9,
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "alpha", "miu", "moa_min", "moa_max"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.alpha = cy.validator(int, alpha, [2, 10], "alpha")
        self.miu = cy.validator(float, miu, [0.1, 2.0], "miu")
        self.moa_min = cy.validator(float, moa_min, (0, 0.41), "moa_min")
        self.moa_max = cy.validator(float, moa_max, (0.41, 1.0), "moa_max")

    def _evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        moa = self.moa_min + epoch * (
                (self.moa_max - self.moa_min) / self.epoch
        )  # Eq. 2
        mop = 1 - ((<object>(epoch + 1)) ** (<object>(1.0 / self.alpha))) / (
                (<object>self.epoch) ** (<object>(1.0 / self.alpha))
        )  # Eq. 4
        g = np.array(self.g_best_x())
        R = self.generator.random((n, d, 3))  # per agent and dimension: r1, r2, r3
        r1, r2, r3 = R[..., 0], R[..., 1], R[..., 2]
        span = (self.problem.bounds.up - self.problem.bounds.low) * self.miu + self.problem.bounds.low
        pos = np.where(
            r1 > moa,  # Exploration phase
            np.where(r2 < 0.5, g / (mop + self.EPSILON) * span, g * mop * span),
            np.where(r3 < 0.5, g - mop * span, g + mop * span),  # Exploitation phase
        )
        cand.X[:] = self._correct_solution(pos)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)
