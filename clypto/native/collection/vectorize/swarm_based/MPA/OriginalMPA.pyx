#!/usr/bin/env python
# Created by "Thieu" at 17:28, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalMPA(LegacyNativeOptimizer):
    """
    The developed version: Marine Predators Algorithm (MPA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417420302025
        2. https://www.mathworks.com/matlabcentral/fileexchange/74578-marine-predators-algorithm-mpa

    Notes:
        1. To use the original paper, set the training mode = "swarm"
        2. They update the whole population at the same time before update the fitness
        3. Two variables that they consider it as constants which are FADS = 0.2 and P = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import MPA    >>> import numpy as np
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
    >>> model = MPA.OriginalMPA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Mirjalili, S., & Gandomi, A. H. (2020).
    Marine Predators Algorithm: A nature-inspired metaheuristic. Expert systems with applications, 152, 113377.
    """

    cdef public object FADS
    cdef public object P

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
        self.FADS = 0.2
        self.P = 0.5

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        CF = (1 - epoch / self.epoch) ** (2 * epoch / self.epoch)
        RL = self.get_levy_flight_step(beta=1.5, multiplier=0.05, size=(n, d), case=-1)
        RB = rng.standard_normal((n, d))
        per1 = rng.permutation(n)
        per2 = rng.permutation(n)
        R = rng.random((n, d))
        if epoch < self.epoch / 3:  # Phase 1 (Eq.12)
            pos = X + self.P * R * (RB * (g - RB * X))
        elif self.epoch / 3 < epoch < 2 * self.epoch / 3:  # Phase 2 (Eqs. 13 & 14)
            second = (np.arange(n) > n / 2)[:, None]
            pos = np.where(second, g + self.P * CF * (RB * (RB * g - X)), X + self.P * R * (RL * (g - RL * X)))
        else:  # Phase 3 (Eq. 15)
            pos = g + self.P * CF * (RL * (RL * g - X))
        pos = self.correct_solution(pos)
        # eddy formation and FADs effect
        fads = (rng.random(n) < self.FADS)[:, None]
        u = np.where(rng.random((n, d)) < self.FADS, 1, 0)
        r = rng.random((n, 1))
        pos_fads = pos + CF * (lb + rng.random((n, d)) * (ub - lb)) * u
        pos_shift = pos + (self.FADS * (1 - r) + r) * (X[per1] - X[per2])
        ops.step(self, np.where(fads, pos_fads, pos_shift))
