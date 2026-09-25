#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalGSKA(AgentListOptimizer):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Links:
        1. https://doi.org/10.1007/s13042-019-01053-x

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kf (float): [0.3, 0.8], knowledge factor that controls the total amount of gained and shared knowledge added from others to the current individual during generations, default = 0.5
        + kr (float): [0.5, 0.95], knowledge ratio, default = 0.9
        + kg (int): [3, 20], number of generations effect to D-dimension, default = 5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import GSKA    >>> import numpy as np
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
    >>> model = GSKA.OriginalGSKA(epoch=1000, pop_size=50, pb = 0.1, kf = 0.5, kr = 0.9, kg = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Mohamed, A.K., 2020. Gaining-sharing knowledge based algorithm for solving
    optimization problems: a novel nature-inspired algorithm. International Journal of Machine Learning and Cybernetics, 11(7), pp.1501-1529.
    """

    cdef public object pb
    cdef public object kf
    cdef public object kr
    cdef public object kg

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pb: float = 0.1,
        kf: float = 0.5,
        kr: float = 0.9,
        kg: int = 5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kf (float): knowledge factor that controls the total amount of gained and shared knowledge added
                        from others to the current individual during generations, default = 0.5
            kr (float): knowledge ratio, default = 0.9
            kg (int): Number of generations effect to D-dimension, default = 5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pb", "kf", "kr", "kg"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.kf = cy.validator(float, kf, (0, 1.0), "kf")
        self.kr = cy.validator(float, kr, (0, 1.0), "kr")
        self.kg = cy.validator(int, kg, [1, 1 + int(epoch / 2)], "kg")

    def evolve_agents(self, epoch):
        dd = int(self.problem.n_dims * (1 - epoch / self.epoch) ** self.kg)
        pop_new = []
        for idx in range(0, self.pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == self.pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1
            # The random individual is for all dimension values
            rand_idx = self.generator.choice(
                list(set(range(0, self.pop_size)) - {previ, idx, nexti})
            )
            pos_new = self.objs[idx].solution.copy()

            for j in range(0, self.problem.n_dims):
                if j < dd:  # junior gaining and sharing
                    if self.generator.uniform() <= self.kr:
                        if self.compare_target(
                                self.objs[rand_idx].target,
                                self.objs[idx].target,
                                self.problem.minmax,
                        ):
                            pos_new[j] = self.objs[idx].solution[j] + self.kf * (
                                    self.objs[previ].solution[j]
                                    - self.objs[nexti].solution[j]
                                    + self.objs[rand_idx].solution[j]
                                    - self.objs[idx].solution[j]
                            )
                        else:
                            pos_new[j] = self.objs[idx].solution[j] + self.kf * (
                                    self.objs[previ].solution[j]
                                    - self.objs[nexti].solution[j]
                                    + self.objs[idx].solution[j]
                                    - self.objs[rand_idx].solution[j]
                            )
                else:  # senior gaining and sharing
                    if self.generator.uniform() <= self.kr:
                        id1 = int(self.pb * self.pop_size)
                        id2 = int(id1 + self.pop_size * (1 - 2 * self.pb))
                        rand_best = self.generator.choice(
                            list(set(range(0, id1)) - {idx})
                        )
                        rand_worst = self.generator.choice(
                            list(set(range(id2, self.pop_size)) - {idx})
                        )
                        rand_mid = self.generator.choice(
                            list(set(range(id1, id2)) - {idx})
                        )
                        if self.compare_target(
                                self.objs[rand_mid].target,
                                self.objs[idx].target,
                                self.problem.minmax,
                        ):
                            pos_new[j] = self.objs[idx].solution[j] + self.kf * (
                                    self.objs[rand_best].solution[j]
                                    - self.objs[rand_worst].solution[j]
                                    + self.objs[rand_mid].solution[j]
                                    - self.objs[idx].solution[j]
                            )
                        else:
                            pos_new[j] = self.objs[idx].solution[j] + self.kf * (
                                    self.objs[rand_best].solution[j]
                                    - self.objs[rand_worst].solution[j]
                                    + self.objs[idx].solution[j]
                                    - self.objs[rand_mid].solution[j]
                            )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    agent, self.objs[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
