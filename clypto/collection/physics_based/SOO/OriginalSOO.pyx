#!/usr/bin/env python
# Created by "Thieu" at 22:08, 28/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSOO(LegacyNativeOptimizer):
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
    >>> from clypto.collection.physics_based import SOO    >>> import numpy as np
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
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void initialize_variables(self):
        self.initial_period = 3

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        g_best = np.array(self.g_best_x())
        # Update period and angular frequency
        caf = 2 * np.pi / (self.initial_period + 0.001 * epoch)

        # Update scaling factor
        scaler = 2 * (1.0 - epoch / self.epoch)

        # Update positions of star oscillators (own row only: fully batched)
        R = self.generator.random((n, 3, d))
        r1, r2, r3 = R[:, 0], R[:, 1], R[:, 2]
        X = pop.X
        # Calculate oscillation positions
        osc1 = scaler * (caf * r1 - 1) * (X - np.abs(r1 * np.sin(r2) * np.abs(r3 * g_best)))
        osc1_pos = g_best - r1 * r3 * osc1
        osc2 = scaler * (caf * r1 - 1) * (X - np.abs(r1 * np.cos(r2) * np.abs(r3 * g_best)))
        osc2_pos = g_best - r2 * r3 * osc2
        cand.X[:] = self.correct_solution(r3 * (osc1_pos + osc2_pos) / 2)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)

        # Get top 3 stars
        best3 = pop.X[self.sorted_order(pop)[:3]]
        cand = pop.empty_like()

        # Perform oscillatory movement update (reads the rows updated before it)
        Xp = pop.X
        for idx in range(self.pop_size):
            # Average of top star positions
            avg3 = np.mean(best3, axis=0)

            # Select 3 random indices different from current
            i1, i2, i3 = self.generator.choice(list(set(range(self.pop_size)) - {idx}), size=3, replace=False)

            # Generate new position based on oscillatory movement
            rf = self.generator.random()
            pos_new = avg3 + 0.5 * (
                    np.sin(rf * np.pi) * (Xp[i1] - Xp[i2])
                    + np.cos((1 - rf) * np.pi) * (Xp[i1] - Xp[i3])
            )
            ## Probabilistic update
            pos_new = np.where(self.generator.random(size=d) <= 0.5, pos_new, Xp[idx])
            # Apply boundary constraints
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
