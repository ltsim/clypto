#!/usr/bin/env python
# Created by "Thieu" at 23:50, 28/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalEAO(LegacyNativeOptimizer):
    """
    The original version of: Enzyme Action Optimizer (EAO)

    Notes:
        + This algorithm used 3 fitness calculations for each update enzyme. Therefor, it is slower 3 times than other algorithms.

    Links:
        1. https://mathworks.com/matlabcentral/fileexchange/170296-enzyme-action-optimizer-a-novel-bio-inspired-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import EAO    >>> import numpy as np
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
    >>> model = EAO.OriginalEAO(epoch=1000, pop_size=50, p_m=0.01, n_elites=2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Rodan, A., Al-Tamimi, A. K., Al-Alnemer, L., Mirjalili, S., & Tiňo, P. (2025).
    Enzyme action optimizer: a novel bio-inspired optimization algorithm. The Journal of Supercomputing, 81(5), 686.
    """

    cdef public object ec

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        ec: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Initialize the algorithm components.

        Args:
            epoch: Maximum number of iterations, default = 10000
            pop_size: Number of population size, default = 100
            ec: Enzyme Concentration, default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "ec"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.ec = cy.validator(float, ec, [0.0, 100], "ec")

    cdef void evolve(self, int epoch_c):
        # Each enzyme reads the enzymes updated before it (two random ones), so the loop is
        # sequential on the buffer rows; the three candidates of an enzyme are evaluated together.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cands
        cdef Py_ssize_t idx, n = pop.n
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        # Adaptation Factor - tăng dần theo thời gian
        AF = np.sqrt(epoch_c / self.epoch)
        d = self.problem.n_dims

        # Handle each enzyme
        for idx in range(self.pop_size):
            # 1. Update FirstSubstratePosition
            r1 = self.generator.random(size=d)
            pos1 = (g_best - Xp[idx]) + r1 * np.sin(AF * Xp[idx])
            pos1 = self.correct_solution(pos1)

            # 2. Select 2 randoms
            j1, j2 = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), size=2, replace=False)

            ## Candidate A: vector-valued random factors
            scA1 = self.ec + (1 - self.ec) * self.generator.random(size=d)
            exA = AF * (self.ec + (1 - self.ec) * self.generator.random(size=d))
            posA = Xp[idx] + scA1 * (Xp[j1] - Xp[j2]) + exA * (g_best - Xp[idx])
            posA = self.correct_solution(posA)

            ## Candidate B: scalar random factors
            scB1 = self.ec + (1 - self.ec) * self.generator.random()
            exB = AF * (self.ec + (1 - self.ec) * self.generator.random())
            posB = Xp[idx] + scB1 * (Xp[j1] - Xp[j2]) + exB * (g_best - Xp[idx])
            posB = self.correct_solution(posB)

            # the best of [current, agent1, agentA, agentB] (the classic argsort tie rule)
            cands = self.new_population(np.array([pos1, posA, posB]))
            order = np.argsort(np.concatenate([[pop.F[idx]], cands.F]))
            if self.problem.minmax == "max":
                order = order[::-1]
            if order[0] > 0:
                pop.buf[idx] = cands.buf[order[0] - 1]
