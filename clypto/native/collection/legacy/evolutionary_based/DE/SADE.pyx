#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class SADE(LegacyOptimizer):
    """
    The original version of: Self-Adaptive Differential Evolution (SADE)

    Links:
        1. https://doi.org/10.1109/CEC.2005.1554904

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.SADE(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for
    numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self._set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def _initialize_variables(self):
        self.loop_probability = 50
        self.loop_cr = 5
        self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0
        self.crm = 0.5
        self.p1 = 0.5
        self.dyn_list_cr = list()

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        list_probability = []
        list_cr = []
        for idx in range(0, self.pop_size):
            ## Calculate adaptive parameter cr and f
            cr = self.generator.normal(self.crm, 0.1)
            cr = np.clip(cr, 0, 1)
            list_cr.append(cr)
            while True:
                f = self.generator.normal(0.5, 0.3)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            id1, id2, id3 = self.generator.choice(
                list(set(range(0, self.pop_size)) - {idx}), 3, replace=False
            )
            if self.generator.random() < self.p1:
                x_new = self.pop[id1].solution + f * (
                    self.pop[id2].solution - self.pop[id3].solution
                )
                pos_new = np.where(
                    self.generator.random(self.problem.n_dims) < cr,
                    x_new,
                    self.pop[idx].solution,
                )
                j_rand = self.generator.integers(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self._correct_solution(pos_new)
                list_probability.append(True)
            else:
                x_new = (
                    self.pop[idx].solution
                    + f * (self.g_best.solution - self.pop[idx].solution)
                    + f * (self.pop[id1].solution - self.pop[id2].solution)
                )
                pos_new = np.where(
                    self.generator.random(self.problem.n_dims) < cr,
                    x_new,
                    self.pop[idx].solution,
                )
                j_rand = self.generator.integers(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self._correct_solution(pos_new)
                list_probability.append(False)
            agent = self._generate_empty_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self._get_target(pos_new)
        pop = self._update_target_for_population(pop)
        for idx in range(0, self.pop_size):
            if list_probability[idx]:
                if self._compare_target(
                    pop[idx].target, self.pop[idx].target, self.problem.sense
                ):
                    self.ns1 += 1
                    self.pop[idx] = pop[idx].copy()
                else:
                    self.nf1 += 1
            else:
                if self._compare_target(
                    pop[idx].target, self.pop[idx].target, self.problem.sense
                ):
                    self.ns2 += 1
                    self.dyn_list_cr.append(list_cr[idx])
                    self.pop[idx] = pop[idx].copy()
                else:
                    self.nf2 += 1
        # Update cr and p1
        if epoch / self.loop_cr == 0:
            self.crm = np.mean(self.dyn_list_cr)
            self.dyn_list_cr = list()
        if epoch / self.loop_probability == 0:
            self.p1 = (
                self.ns1
                * (self.ns2 + self.nf2)
                / (self.ns2 * (self.ns1 + self.nf1) + self.ns1 * (self.ns2 + self.nf2))
            )
            self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0
