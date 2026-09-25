#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent cimport LegacyNativeAgent
from clypto.optimizer._native.target cimport NativeTarget


cdef class SADE(LegacyNativeOptimizer):
    """
    The original version of: Self-Adaptive Differential Evolution (SADE)

    Links:
        1. https://doi.org/10.1109/CEC.2005.1554904

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.SADE(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for
    numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE.
    """

    cdef public object loop_probability
    cdef public object loop_cr
    cdef public object ns1
    cdef public object ns2
    cdef public object nf1
    cdef public object nf2
    cdef public object crm
    cdef public object p1
    cdef public object dyn_list_cr
    cdef public object objs

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
        self.loop_probability = 50
        self.loop_cr = 5
        self.ns1 = self.ns2 = self.nf1 = self.nf2 = 0
        self.crm = 0.5
        self.p1 = 0.5
        self.dyn_list_cr = list()

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        self.objs = ops.agents_of(self.pop)
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
                x_new = self.objs[id1].solution + f * (
                    self.objs[id2].solution - self.objs[id3].solution
                )
                pos_new = np.where(
                    self.generator.random(self.problem.n_dims) < cr,
                    x_new,
                    self.objs[idx].solution,
                )
                j_rand = self.generator.integers(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.correct_solution(pos_new)
                list_probability.append(True)
            else:
                x_new = (
                    self.objs[idx].solution
                    + f * (self.g_best.solution - self.objs[idx].solution)
                    + f * (self.objs[id1].solution - self.objs[id2].solution)
                )
                pos_new = np.where(
                    self.generator.random(self.problem.n_dims) < cr,
                    x_new,
                    self.objs[idx].solution,
                )
                j_rand = self.generator.integers(0, self.problem.n_dims)
                pos_new[j_rand] = x_new[j_rand]
                pos_new = self.correct_solution(pos_new)
                list_probability.append(False)
            agent = LegacyNativeAgent(pos_new, None)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].target = self.get_target(pos_new)
        pop = ops.update_targets(self, pop)
        for idx in range(0, self.pop_size):
            if list_probability[idx]:
                if self.compare_fitness(pop[idx].target.fitness, self.objs[idx].target.fitness, self.problem.minmax):
                    self.ns1 += 1
                    self.objs[idx] = pop[idx].copy()
                else:
                    self.nf1 += 1
            else:
                if self.compare_fitness(pop[idx].target.fitness, self.objs[idx].target.fitness, self.problem.minmax):
                    self.ns2 += 1
                    self.dyn_list_cr.append(list_cr[idx])
                    self.objs[idx] = pop[idx].copy()
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
        self.pop = ops.population_of(self.pop, self.objs)
