#!/usr/bin/env python
# Created by "Thieu" at 10:06, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class HI_WOA(VectorizeOptimizer):
    """
    The original version of: Hybrid Improved Whale Optimization Algorithm (HI-WOA)

    Links:
        1. https://ieenp.explore.ieee.org/document/8900003

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + feedback_max (int): maximum iterations of each feedback, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import WOA    >>> import numpy as np
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
    >>> model = WOA.HI_WOA(epoch=1000, pop_size=50, feedback_max = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Tang, C., Sun, W., Wu, W. and Xue, M., 2019, July. A hybrid improved whale optimization algorithm.
    In 2019 IEEE 15th International Conference on Control and Automation (ICCA) (pp. 362-367). IEEE.
    """

    cdef public object feedback_max
    cdef public object n_changes
    cdef public object dyn_feedback_count

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        feedback_max: int = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            feedback_max (int): maximum iterations of each feedback, default = 10
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "feedback_max"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.feedback_max = cy.validator(int, feedback_max, [2, 2 + int(self.epoch / 2)], "feedback_max")

    def _initialize_variables(self):
        self.n_changes = int(self.pop_size / 2)
        self.dyn_feedback_count = 0

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativePopulation child
        cdef Py_ssize_t idx, n = pop.n
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        a = 2 + 2 * np.cos(np.pi / 2 * (1 + epoch / self.epoch))  # Eq. 8
        for idx in range(0, self.pop_size):
            r = self.generator.random()
            A = 2 * a * r - a
            C = 2 * r
            l = self.generator.uniform(-1, 1)
            p = 0.5
            b = 1
            if self.generator.uniform() < p:
                if np.abs(A) < 1:
                    D = np.abs(C * g_best - Xp[idx])
                    pos_new = g_best - A * D
                else:
                    x_rand = self.problem.generate_solution()
                    D = np.abs(C * x_rand - Xp[idx])
                    pos_new = x_rand - A * D
            else:
                D1 = np.abs(g_best - Xp[idx])
                pos_new = g_best + np.exp(b * l) * np.cos(2 * np.pi * l) * D1
            ops.commit(self, pop, cand, idx, self._correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
        ## Feedback Mechanism
        current_best = pop.F[self.sorted_order(pop)[0]]
        if current_best == gb_fit:
            self.dyn_feedback_count += 1
        else:
            self.dyn_feedback_count = 0
        if self.dyn_feedback_count >= self.feedback_max:
            idx_list = self.generator.choice(range(0, self.pop_size), self.n_changes, replace=False)
            child = self.generate_population(self.n_changes)
            pop.buf[idx_list] = child.buf
