#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class ProbBeesA(VectorizeOptimizer):
    """
    The original version of: Probabilistic Bees Algorithm (P-BeesA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + recruited_bee_ratio (float): percent of bees recruited, default = 0.1
        + dance_factor (tuple, list): (radius, reduction) - Bees Dance Radius, default=(0.1, 0.99)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import BeesA    >>> import numpy as np
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
    >>> model = BeesA.ProbBeesA(epoch=1000, pop_size=50, recruited_bee_ratio = 0.1, dance_radius = 0.1, dance_reduction = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T. and Castellani, M., 2015. A comparative study of the Bees Algorithm as a tool for
    function optimisation. Cogent Engineering, 2(1), p.1091540.
    """

    cdef public object recruited_bee_ratio
    cdef public object dance_radius
    cdef public object dance_reduction
    cdef public object dyn_radius
    cdef public object recruited_bee_count

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        recruited_bee_ratio: float = 0.1,
        dance_radius: float = 0.1,
        dance_reduction: float = 0.99,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float): percent of bees recruited, default = 0.1
            dance_radius (float): Bees Dance Radius, default=0.1
            dance_reduction (float): Bees Dance Radius Reduction Rate, default=0.99
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "recruited_bee_ratio",
                "dance_radius",
                "dance_reduction",
            ],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.recruited_bee_ratio = cy.validator(float, recruited_bee_ratio, (0, 1.0), "recruited_bee_ratio")
        self.dance_radius = cy.validator(float, dance_radius, (0, 1.0), "dance_radius")
        self.dance_reduction = cy.validator(float, dance_reduction, (0, 1.0), "dance_reduction")
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, fresh
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        fit = 1.0 / (np.array(pop.F) + self.EPSILON)
        d_fit = fit / np.mean(fit)
        reject_prob = np.select([d_fit < 0.9, d_fit < 0.95, d_fit < 1.15], [0.6, 0.2, 0.05], default=0.0)
        accepted = rng.random(n) >= reject_prob
        bees = np.clip(np.ceil(d_fit * self.recruited_bee_count).astype(int), 2, n)
        counts = np.where(accepted, bees, 0)
        parent = np.repeat(np.arange(n), counts)
        pos = np.array(pop.X[parent])
        pos[np.arange(len(parent)), rng.integers(0, d, size=len(parent))] += self.dyn_radius * rng.uniform(-1, 1, len(parent))
        cand = pop.take(parent)
        cand.X[:] = self._correct_solution(pos)
        self.evaluate(cand, 0, len(parent))
        ops.scatter(self, cand, parent)
        # rejected sites are abandoned: their bees scout new random sources
        lost = np.flatnonzero(~accepted)
        if len(lost):
            fresh = pop.take(lost)
            fresh.X[:] = self.problem.bounds.low + rng.random((len(lost), d)) * (self.problem.bounds.up - self.problem.bounds.low)
            self.evaluate(fresh, 0, len(lost))
            pop.buf[lost] = fresh.buf
        self.dyn_radius = self.dance_reduction * self.dance_radius
