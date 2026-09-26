#!/usr/bin/env python
# Created by "Thieu" at 22:08, 28/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalSOO(VectorizeOptimizer):
    """
    The original version of: Stellar Oscillation Optimizer (SOO)

    Notes:
        + The MATLAB code in the link below by the author is completely different from the pseudocode in
        the original paper. I don’t understand how the author could write such an incorrect implementation
        and still obtain good results. There are only two possibilities: either the author fabricated the
        results in the paper, or the paper itself is fundamentally flawed.

        + For example, you can see equation number 8 — it involves taking the average of two new positions.
        However, in the code, it is incorrectly implemented as position 1 plus half of position 2.
        Even more concerning is that the pseudocode in the paper is completely different from the actual code.
        The MATLAB coding quality is really poor. In the pseudocode, it states that the fitness should be
        calculated and the global best as well as the top 3 best should be updated — yet this is entirely missing in the code.

        + Therefore, I do not recommend users to use this algorithm, as it lacks integrity between the
        results in the paper and the actual experimental implementation.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/161921-stellar-oscillation-optimizer-meta-heuristic-optimimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import SOO    >>> import numpy as np
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
    >>> model = SOO.OriginalSOO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., & Mirjalili, S. (2025).
    Stellar oscillation optimizer: a nature-inspired metaheuristic optimization algorithm. Cluster Computing, 28(6), 362.
    """

    cdef public object initial_period

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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _initialize_variables(self):
        self.initial_period = 3

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        caf = 2 * np.pi / (self.initial_period + 0.001 * epoch)
        scaler = 2 * (1.0 - epoch / self.epoch)
        R = rng.random((n, 3, d))
        r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
        osc1 = scaler * (caf * r1 - 1) * (X - np.abs(r1 * np.sin(r2) * np.abs(r3 * g)))
        osc2 = scaler * (caf * r1 - 1) * (X - np.abs(r1 * np.cos(r2) * np.abs(r3 * g)))
        ops.step(self, r3 * ((g - r1 * r3 * osc1) + (g - r2 * r3 * osc2)) / 2)
        # second phase: around the mean of the three best agents
        X = pop.X
        avg3 = np.mean(X[self.sorted_order(pop)[:3]], axis=0)
        i = ops.k_others(self, n, 3)
        rf = rng.random((n, 1))
        pos = avg3 + 0.5 * (np.sin(rf * np.pi) * (X[i[:, 0]] - X[i[:, 1]]) + np.cos((1 - rf) * np.pi) * (X[i[:, 0]] - X[i[:, 2]]))
        ops.step(self, np.where(rng.random((n, d)) <= 0.5, pos, X))
