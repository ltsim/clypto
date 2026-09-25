#!/usr/bin/env python
# Created by "Thieu" at 22:37, 03/09/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalSFOA(LegacyNativeOptimizer):
    """
    The original version: Starfish Optimization Algorithm (SFOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/173735-starfish-optimization-algorithm-sfoa

    Notes:
        This algorithm claims to outperform 95 compared algorithms in accuracy and 97 algorithms in efficiency.
        However, it does not present any remarkable equations. Moreover, the provided MATLAB code does not
        include the standard CEC benchmark functions, but only simplified versions of them.
        Users should carefully consider this when validating the algorithm.
        Many new algorithms claim to be superior to other state-of-the-art methods,
        but it is evident that their implementations are often incorrect.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + gp (float): [0., 1] -> better [0.5, 0.7], the probablity for exploration

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SFOA    >>> import numpy as np
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
    >>> model = SFOA.OriginalSFOA(epoch=1000, pop_size=50, gp = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    [1] Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025).
    Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global
    optimization compared with 100 optimizers. Neural Computing and Applications, 37(5), 3641-3683.
    """

    cdef public object gp

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        gp: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            gp (float): the exploration of starfish, default=0.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "gp"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.gp = cy.validator(float, gp, [0, 1.0], "gp")

    cdef void evolve(self, int epoch_c):
        # A different number of draws per agent depending on the branch: candidates are built agent
        # by agent (same draw order) from the unchanged population; evaluation is batched.
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        theta = np.pi / 2 * epoch / self.epoch
        tEO = (self.epoch - epoch) / self.epoch * np.cos(theta)

        if self.generator.random() < self.gp:  # exploration of starfish
            for idx in range(self.pop_size):
                pos_new = Xp[idx].copy()
                if self.problem.n_dims > 5:
                    # for nD is larger than 5
                    jp1 = self.generator.choice(self.problem.n_dims, 5, replace=False)
                    pm = (
                                 2 * self.generator.random(size=self.problem.n_dims) - 1
                         ) * np.pi
                    pos1 = pos_new + pm * (g_best - pos_new) * np.cos(
                        theta
                    )
                    pos2 = pos_new - pm * (g_best - pos_new) * np.sin(
                        theta
                    )
                    pos = np.where(
                        self.generator.random(size=self.problem.n_dims) < self.gp,
                        pos1,
                        pos2,
                    )
                    pos_new[jp1] = pos[jp1]
                    # Boundary check for individual dimension
                    pos_new[jp1] = np.where(
                        (pos_new[jp1] < self.problem.lb[jp1])
                        | (pos_new[jp1] > self.problem.ub[jp1]),
                        Xp[idx][jp1],
                        pos_new[jp1],
                    )
                else:
                    # for nD is not larger than 5
                    jp2 = self.generator.integers(0, self.problem.n_dims)
                    im = self.generator.choice(self.pop_size, 2, replace=False)
                    diff1 = self.pop[im[0]].solution[jp2] - pos_new[jp2]
                    diff2 = self.pop[im[1]].solution[jp2] - pos_new[jp2]
                    rand1 = 2 * self.generator.random() - 1
                    rand2 = 2 * self.generator.random() - 1
                    pos_new[jp2] = tEO * pos_new[jp2] + rand1 * diff1 + rand2 * diff2
                    # Boundary check for individual dimension
                    if (
                            pos_new[jp2] > self.problem.ub[jp2]
                            or pos_new[jp2] < self.problem.lb[jp2]
                    ):
                        pos_new[jp2] = Xp[idx][jp2]
                Xc[idx] = self.correct_solution(pos_new)
        else:  # exploitation of starfish
            df = self.generator.choice(self.pop_size, 5, replace=False)
            # five arms of starfish
            dm1 = g_best - self.pop[df[0]].solution
            dm2 = g_best - self.pop[df[1]].solution
            dm3 = g_best - self.pop[df[2]].solution
            dm4 = g_best - self.pop[df[3]].solution
            dm5 = g_best - self.pop[df[4]].solution
            dm = [dm1, dm2, dm3, dm4, dm5]
            for idx in range(self.pop_size):
                r1, r2 = self.generator.random(size=2)
                kp = self.generator.choice(5, size=2, replace=False)
                pos_new = (
                        Xp[idx] + r1 * dm[kp[0]] + r2 * dm[kp[1]]
                )  # exploitation
                if idx == self.pop_size - 1:  # last individual
                    pos_new = (
                            np.exp(-epoch * self.pop_size / self.epoch)
                            * Xp[idx]
                    )  # regeneration of starfish
                Xc[idx] = self.correct_solution(pos_new)
        # Update population with greedy strategy
        self.evaluate(cand, 0, n)
        ops.greedy(self, cand)
