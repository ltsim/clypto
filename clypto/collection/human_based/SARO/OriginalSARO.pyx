#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 11:16, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.collection.human_based.SARO.DevSARO cimport DevSARO
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalSARO(DevSARO):
    """
    The original version of: Search And Rescue Optimization (SARO)

    Links:
       1. https://doi.org/10.1155/2019/2482543

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + se (float): [0.3, 0.8], social effect, default = 0.5
        + mu (int): [10, 20], maximum unsuccessful search number, default = 15

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import SARO    >>> import numpy as np
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
    >>> model = SARO.OriginalSARO(epoch=1000, pop_size=50, se = 0.5, mu = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shabani, A., Asgarian, B., Gharebaghi, S.A., Salido, M.A. and Giret, A., 2019. A new optimization
    algorithm based on search and rescue operations. Mathematical Problems in Engineering, 2019.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        se: float = 0.5,
        mu: int = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            se (float): social effect, default = 0.5
            mu (int): maximum unsuccessful search number, default = 15
        """
        super().__init__(epoch, pop_size, se, mu, name=name, mode=mode)

    def evolve_agents(self, epoch):
        pop_x = [agent.copy() for agent in self.objs[: self.pop_size]]
        pop_m = [agent.copy() for agent in self.objs[self.pop_size:]]
        pop_new = []
        for idx in range(self.pop_size):
            ## Social Phase
            k = self.generator.choice(list(set(range(0, 2 * self.pop_size)) - {idx}))
            sd = pop_x[idx].solution - self.objs[k].solution
            j_rand = self.generator.integers(0, self.problem.n_dims)
            r1 = self.generator.uniform(-1, 1)

            pos_new = pop_x[idx].solution.copy()
            for j in range(0, self.problem.n_dims):
                if self.generator.uniform() < self.se or j == j_rand:
                    if self.compare_target(
                            self.objs[k].target, pop_x[idx].target, self.problem.minmax
                    ):
                        pos_new[j] = self.objs[k].solution[j] + r1 * sd[j]
                    else:
                        pos_new[j] = pop_x[idx].solution[j] + r1 * sd[j]
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.ub[j]) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(
                    pop_new[idx].target, pop_x[idx].target, self.problem.minmax
            ):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx].copy()
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

        ## Individual phase
        pop = pop_x.copy() + pop_m.copy()
        pop_new = []
        for idx in range(0, self.pop_size):
            k, m = self.generator.choice(
                list(set(range(0, 2 * self.pop_size)) - {idx}), 2, replace=False
            )
            pos_new = pop_x[idx].solution + self.generator.uniform() * (
                    pop[k].solution - pop[m].solution
            )
            for j in range(0, self.problem.n_dims):
                if pos_new[j] < self.problem.lb[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.lb[j]) / 2
                if pos_new[j] > self.problem.ub[j]:
                    pos_new[j] = (pop_x[idx].solution[j] + self.problem.ub[j]) / 2
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self.compare_target(
                    pop_new[idx].target, pop_x[idx].target, self.problem.minmax
            ):
                pop_m[self.generator.integers(0, self.pop_size)] = pop_x[idx]
                pop_x[idx] = pop_new[idx].copy()
                self.dyn_USN[idx] = 0
            else:
                self.dyn_USN[idx] += 1

            if self.dyn_USN[idx] > self.mu:
                pop_x[idx] = self.generate_agent()
                self.dyn_USN[idx] = 0
        self.objs = pop_x + pop_m
