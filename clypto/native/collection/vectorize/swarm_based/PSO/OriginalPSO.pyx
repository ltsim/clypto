#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np

from cython.parallel cimport prange

from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation, np_clip
from clypto.native.collection.vectorize.swarm_based.PSO._base cimport _PSOBase

# Rows x dims below which OpenMP threads cost more than they save.
cdef Py_ssize_t PARALLEL_MIN_WORK = 20000


cdef void _pso_move(
    double[:, ::1] buf, double[:, ::1] cand, const double[:, :, ::1] R,
    const double[::1] g, const double[::1] lb, const double[::1] ub,
    const double[::1] v_min, const double[::1] v_max,
    double w, double c1, double c2, bint clip_velocity,
    Py_ssize_t cX, Py_ssize_t cV, Py_ssize_t cP, Py_ssize_t cand_x,
    Py_ssize_t start, Py_ssize_t stop, Py_ssize_t d,
) noexcept nogil:
    """OriginalPSO/LDW_PSO move with the classic operation order (no FMA)."""
    cdef Py_ssize_t i, j
    cdef double x, v, pos
    for i in prange(start, stop, schedule="static",
                    use_threads_if=(stop - start) * d > PARALLEL_MIN_WORK):
        for j in range(d):
            x = buf[i, cX + j]
            v = (w * buf[i, cV + j] + (c1 * R[i, 0, j]) * (buf[i, cP + j] - x)) \
                + (c2 * R[i, 1, j]) * (g[j] - x)
            if clip_velocity:
                v = np_clip(v, v_min[j], v_max[j])
            buf[i, cV + j] = v
            pos = x + v
            if not (lb[j] <= pos and pos <= ub[j]):
                pos = lb[j] + (ub[j] - lb[j]) * R[i, 2, j]
            cand[i, cand_x + j] = pos



cdef class OriginalPSO(_PSOBase):
    """
    The original version of: Particle Swarm Optimization (PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + w (float): (0., 1.0), Weight min of bird, default = 0.4

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.OriginalPSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, w=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kennedy, J. and Eberhart, R., 1995, November. Particle swarm optimization. In Proceedings of
    ICNN'95-international conference on neural networks (Vol. 4, pp. 1942-1948). IEEE.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        w: float = 0.4,
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
            w: Weight of bird, default = 0.4
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c1", "c2", "w"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.w = cy.validator(float, w, (0, 1.0), "w")

    cdef double weight(self, int epoch):
        return self.w

    cdef bint clips_velocity(self):
        return False

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t start, stop, n = pop.n, d = pop.d
        cdef double w = self.weight(epoch)
        R = self.generator.random((n, 3, d))  # per agent: r1, r2, amend draws
        lb = np.ascontiguousarray(self.problem.lb, dtype=float)
        ub = np.ascontiguousarray(self.problem.ub, dtype=float)
        v_min = np.ascontiguousarray(self.v_min, dtype=float)
        v_max = np.ascontiguousarray(self.v_max, dtype=float)
        for start, stop in self.chunks(n):
            g = np.array(self.g_best_x(), dtype=float)
            _pso_move(pop.view, cand.view, R, g, lb, ub, v_min, v_max,
                      w, self.c1, self.c2, self.clips_velocity(),
                      pop.cX, self.cV, self.cP, cand.cX, start, stop, d)
            cand.X[start:stop] = self.problem.correct_solutions(cand.X[start:stop])
            self.evaluate(cand, start, stop)
            self.accept(cand, start, stop)
