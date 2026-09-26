#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalASO(LegacyNativeOptimizer):
    """
    The original version of: Atom Search Optimization (ASO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2018.08.030
        2. https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (int): Depth weight, default = 10, depend on the problem
        + beta (float): Multiplier weight, default = 0.2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import ASO    >>> import numpy as np
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
    >>> model = ASO.OriginalASO(epoch=1000, pop_size=50, alpha = 50, beta = 0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2019. Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem. Knowledge-Based Systems, 163, pp.283-304.
    """

    cdef public int alpha
    cdef public double beta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: int = 10,
        beta: float = 0.2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): [2, 20], Depth weight, default = 10
            beta (float): [0.1, 1.0], Multiplier weight, default = 0.2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "alpha", "beta"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.alpha = cy.validator(int, alpha, [1, 100], "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("V", d), ("M", 1)]  # velocity and mass of each atom

    cdef void init_fields(self, NativePopulation pop):
        pop.field("V")[:] = self.generator.uniform(self.problem.lb, self.problem.ub, (pop.n, pop.d))
        pop.field("M")[:] = 0.0

    cdef object amend_solution(self, object solution):
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(condition, solution, rand_pos)

    def update_mass__(self, NativePopulation pop):
        list_fit = np.array(pop.F)
        list_fit = np.exp(
            -(list_fit - np.max(list_fit))
            / (np.max(list_fit) - np.min(list_fit) + self.EPSILON)
        )
        list_fit = list_fit / np.sum(list_fit)
        pop.field("M")[:, 0] = list_fit

    def find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * np.sin(iteration / self.epoch * np.pi / 2)
        rsmax = 1.24
        if radius / average_dist < rsmin:
            rs = rsmin
        else:
            if radius / average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7))
        return potential

    def acceleration__(self, NativePopulation pop, g_best, iteration):
        cdef Py_ssize_t idx, n = pop.n
        eps = 2 ** (-52)
        self.update_mass__(pop)
        mass = pop.field("M")[:, 0]
        X = pop.X
        G = np.exp(-20.0 * iteration / self.epoch)
        k_best = (
            int(self.pop_size - (self.pop_size - 2) * (<object>(iteration / self.epoch)) ** 0.5)
            + 1
        )
        # k_best atoms with the largest (min problems) / smallest (max problems) mass
        k_best_idx = sorted(range(n), key=lambda i: mass[i], reverse=(self.problem.minmax == "min"))[:k_best]
        mk_average = np.mean(np.array([X[i] for i in k_best_idx]))
        acc_list = np.zeros((self.pop_size, self.problem.n_dims))
        for idx in range(0, self.pop_size):
            dist_average = np.linalg.norm(X[idx] - mk_average)
            temp = np.zeros((self.problem.n_dims))
            for atom in k_best_idx:
                # calculate LJ-potential
                radius = np.linalg.norm(X[idx] - X[atom])
                potential = self.find_LJ_potential__(iteration, dist_average, radius)
                temp += (
                    potential
                    * self.generator.uniform(0, 1, self.problem.n_dims)
                    * ((X[atom] - X[idx]) / (radius + eps))
                )
            temp = self.alpha * temp + self.beta * (g_best - X[idx])
            # calculate acceleration
            acc = G * temp / mass[idx]
            acc_list[idx] = acc
        return acc_list

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n
        # g_best is the best row of this epoch's start (its row is refreshed below, like the classic agent)
        best_row = self.sorted_order(pop)[0]
        # Calculate acceleration (also refreshes the masses of the population).
        atom_acc_list = self.acceleration__(pop, np.array(self.g_best_x()), iteration=epoch)
        g_row = pop.buf[best_row].copy()
        g_fit = float(pop.F[best_row])
        cand.buf[:] = pop.buf
        # Update velocity based on random dimensions and position of global best. The velocity is
        # computed but never stored (as in the classic version); amend_solution draws one
        # uniform block per agent, right after the agent's random(d) draw.
        R = self.generator.random((n, 2, pop.d))
        velocity = R[:, 0] * pop.field("V") + atom_acc_list
        pos_new = pop.X + velocity
        lb, ub = self.problem.lb, self.problem.ub
        pos_new = np.where(np.logical_and(lb <= pos_new, pos_new <= ub), pos_new, lb + (ub - lb) * R[:, 1])
        cand.X[:] = self.problem.correct_solutions(pos_new)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)
        current_best = self.sorted_order(cand)[0]
        if self.compare_fitness(g_fit, cand.F[current_best], self.problem.minmax):
            pop.buf[self.generator.integers(0, self.pop_size)] = g_row
