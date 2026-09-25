#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalGJO(LegacyNativeOptimizer):
    """
    The original version of: Golden jackal optimization (GJO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S095741742200358X
        2. https://www.mathworks.com/matlabcentral/fileexchange/108889-golden-jackal-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import GJO    >>> import numpy as np
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
    >>> model = GJO.OriginalGJO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chopra, N., & Ansari, M. M. (2022). Golden jackal optimization: A novel nature-inspired
    optimizer for engineering applications. Expert Systems with Applications, 198, 116924.
    """

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

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        E1 = 1.5 * (1.0 - (epoch / self.epoch))
        RL = self.get_levy_flight_step(
            beta=1.5,
            multiplier=0.05,
            size=(self.pop_size, self.problem.n_dims),
            case=-1,
        )
        order = self.sorted_order(pop)
        male = np.array(pop.X[order[0]])
        female = np.array(pop.X[order[1]])
        Xp = np.array(pop.X)
        # one draw per (agent, dimension), agent by agent
        E = E1 * (2 * self.generator.random((n, d)) - 1)
        exploit = np.abs(E) < 1
        t1 = np.where(exploit, np.abs(RL * male - Xp), np.abs(male - RL * Xp))
        t2 = np.where(exploit, np.abs(RL * female - Xp), np.abs(female - RL * Xp))
        male_pos = male - E * t1
        female_pos = female - E * t2
        pos_new = (male_pos + female_pos) / 2
        Xc = cand.X
        for idx in range(n):
            Xc[idx] = self.correct_solution(pos_new[idx])
        # every agent is replaced by its candidate
        self.evaluate(cand, 0, n)
        self.pop = cand
