#!/usr/bin/env python
# Created by "Thieu" at 12:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.bio_based.SBO.DevSBO cimport DevSBO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalSBO(DevSBO):
    """
    The original version of: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006
        2. https://www.mathworks.com/matlabcentral/fileexchange/62009-satin-bowerbird-optimizer-sbo-2017

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 3.0] -> better [0.5, 0.99], the greatest step size
        + p_m (float): (0, 1.0) -> better [0.01, 0.2], mutation probability
        + psw (float): (0, 1.0) -> better [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import SBO    >>> import numpy as np
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
    >>> model = SBO.OriginalSBO(epoch=1000, pop_size=50, alpha = 0.9, p_m=0.05, psw = 0.02)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Moosavi, S.H.S. and Bardsiri, V.K., 2017. Satin bowerbird optimizer: A new optimization algorithm
    to optimize ANFIS for software development effort estimation. Engineering Applications of Artificial Intelligence, 60, pp.1-15.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.94,
        p_m: float = 0.05,
        psw: float = 0.02,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            p_m (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
        """
        super().__init__(epoch, pop_size, alpha, p_m, psw, name=name, mode=mode)

    def roulette_wheel_selection__(self, fitness_list: list | np.ndarray | None = None) -> int:
        r = self.generator.uniform()
        c = np.cumsum(fitness_list)
        f = np.where(r < c)[0][0]
        return f

    cdef void evolve(self, int epoch_c):
        # The classic sequential path only refreshes the *targets* of the population (with the
        # candidates' fitness) and never moves it; swarm modes replace the population.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativeTarget tar
        cdef Py_ssize_t idx, jdx, n = pop.n
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp, Xc = pop.X, cand.X
        g_best = np.array(self.g_best_x())
        # (percent of the difference between the upper and lower limit (Eq. 7))
        self.sigma = self.psw * (self.problem.ub - self.problem.lb)
        ## Calculate the probability of bowers using Eqs. (1) and (2)
        fx_list = np.array(pop.F)
        fit_list = fx_list.copy()
        for idx in range(0, self.pop_size):
            if fx_list[idx] < 0:
                fit_list[idx] = 1.0 + np.abs(fx_list[idx])
            else:
                fit_list[idx] = 1.0 / (1.0 + np.abs(fx_list[idx]))
        fit_sum = np.sum(fit_list)
        ## Calculating the probability of each bower
        prob_list = fit_list / fit_sum
        for idx in range(0, self.pop_size):
            pos_new = Xp[idx].copy()
            for jdx in range(0, self.problem.n_dims):
                ### Select a bower using roulette wheel
                rdx = self.roulette_wheel_selection__(prob_list)
                ### Calculating Step Size
                lamda = self.alpha / (1 + prob_list[rdx])
                pos_new[jdx] = Xp[idx][jdx] + lamda * ((Xp[rdx][jdx] + g_best[jdx]) / 2 - Xp[idx][jdx])
                ### Mutation
                if self.generator.uniform() < self.p_m:
                    pos_new[jdx] = Xp[idx][jdx] + self.generator.normal(0, 1) * self.sigma[jdx]
            pos_new = self.correct_solution(pos_new)
            if swarm:
                Xc[idx] = pos_new
            else:
                tar = self.get_target(pos_new)
                pop.O[idx] = tar.objectives
                pop.F[idx] = tar.fitness
        if swarm:
            self.evaluate(cand, 0, n)
            self.pop = cand
