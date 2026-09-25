#!/usr/bin/env python
# Created by "Thieu" at 12:01, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalALO(LegacyNativeOptimizer):
    """
    The original version of: Ant Lion Optimizer (ALO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/49920-ant-lion-optimizer-alo
        2. https://dx.doi.org/10.1016/j.advengsoft.2015.01.010

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ALO    >>> import numpy as np
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
    >>> model = ALO.OriginalALO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mirjalili, S., 2015. The ant lion optimizer. Advances in engineering software, 83, pp.80-98.
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
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def random_walk_antlion__(self, solution, current_epoch):
        I = 1  # I is the ratio in Equations (2.10) and (2.11)
        if current_epoch > self.epoch / 10:
            I = 1 + 100 * (current_epoch / self.epoch)
        if current_epoch > self.epoch / 2:
            I = 1 + 1000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * (3 / 4):
            I = 1 + 10000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.9:
            I = 1 + 100000 * (current_epoch / self.epoch)
        if current_epoch > self.epoch * 0.95:
            I = 1 + 1000000 * (current_epoch / self.epoch)

        # Decrease boundaries to converge towards antlion
        lb = self.problem.lb / I  # Equation (2.10) in the paper
        ub = self.problem.ub / I  # Equation (2.10) in the paper

        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
        if self.generator.random() < 0.5:
            lb = lb + solution  # Equation(2.8) in the paper
        else:
            lb = -lb + solution
        if self.generator.random() < 0.5:
            ub = ub + solution  # Equation(2.9) in the paper
        else:
            ub = -ub + solution

        # n random walks (one per dimension), normalized according to lb and ub
        X = np.cumsum(2 * (self.generator.random((self.problem.n_dims, self.epoch)) > 0.5) - 1, axis=1)
        a = np.min(X, axis=1)[:, None]
        b = np.max(X, axis=1)[:, None]
        c = lb[:, None]  # [a b] - -->[c d]
        d = ub[:, None]
        return ((X - a) * (d - c)) / (b - a) + c  # Equation(2.7) in the paper

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        Xp, Xc = pop.X, cand.X
        list_fitness = np.array(pop.F)
        g_best = self.current_g_best()
        elite_x = np.array(g_best.solution)
        for idx in range(0, self.pop_size):
            # Select ant lions based on their fitness (the better anlion the higher chance of catching ant)
            rolette_index = self.get_index_roulette_wheel_selection(list_fitness)
            # RA is the random walk around the selected antlion by rolette wheel
            RA = self.random_walk_antlion__(Xp[rolette_index], epoch)
            # RE is the random walk around the elite (the best antlion so far)
            RE = self.random_walk_antlion__(elite_x, epoch)
            temp = (RA[:, epoch - 1] + RE[:, epoch - 1]) / 2  # Equation(2.13) in the paper
            # Bound checking (bring back the antlions of ants inside search space if they go beyonds the boundaries
            Xc[idx] = self.correct_solution(temp)
        self.evaluate(cand, 0, n)
        # Update antlion positions and fitnesses based on the ants (if an ant becomes fitter than an antlion
        # we assume it was caught by the antlion and the antlion update goes to its position to build the trap)
        both = pop.concat(cand)
        pop = self.pop = both.take(self.sorted_order(both)[:n])
        # Keep the elite in the population
        ops.set_row(pop, n - 1, elite_x, g_best.target)
