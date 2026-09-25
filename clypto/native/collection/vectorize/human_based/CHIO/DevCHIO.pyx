#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 19:24, 09/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.human_based.CHIO.OriginalCHIO cimport OriginalCHIO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class DevCHIO(OriginalCHIO):
    """
    The developed version of: Coronavirus Herd Immunity Optimization (CHIO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + brr (float): [0.05, 0.2], Basic reproduction rate, default=0.15
        + max_age (int): [5, 20], Maximum infected cases age, default=10

    Examples
    ~~~~~~~~
    >>> from clypto.collection.human_based import CHIO    >>> import numpy as np
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
    >>> model = CHIO.DevCHIO(epoch=1000, pop_size=50, brr = 0.15, max_age = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        brr: float = 0.15,
        max_age: int = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            brr (float): Basic reproduction rate, default=0.15
            max_age (int): Maximum infected cases age, default=10
        """
        super().__init__(epoch, pop_size, brr, max_age, name=name, mode=mode)

    def evolve_agents(self, epoch):
        pop_new = []
        is_corona_list = [
                             False,
                         ] * self.pop_size
        for i in range(0, self.pop_size):
            pos_new = self.objs[i].solution.copy()
            for j in range(0, self.problem.n_dims):
                rand = self.generator.uniform()
                if rand < (1.0 / 3) * self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 1
                    )  # Infected list
                    if idx_candidates[0].size == 0:
                        rand_choice = self.generator.choice(
                            range(0, self.pop_size),
                            int(0.33 * self.pop_size),
                            replace=False,
                        )
                        self.immunity_type_list[rand_choice] = 1
                        idx_candidates = np.where(self.immunity_type_list == 1)
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.objs[i].solution[j] + self.generator.uniform() * (
                            self.objs[i].solution[j] - self.objs[idx_selected].solution[j]
                    )
                    is_corona_list[i] = True
                elif (1.0 / 3) * self.brr <= rand < (2.0 / 3) * self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 0
                    )  # Susceptible list
                    if idx_candidates[0].size == 0:
                        rand_choice = self.generator.choice(
                            range(0, self.pop_size),
                            int(0.33 * self.pop_size),
                            replace=False,
                        )
                        self.immunity_type_list[rand_choice] = 0
                        idx_candidates = np.where(self.immunity_type_list == 0)
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.objs[i].solution[j] + self.generator.uniform() * (
                            self.objs[i].solution[j] - self.objs[idx_selected].solution[j]
                    )
                elif (2.0 / 3) * self.brr <= rand < self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 2
                    )  # Immunity list
                    fit_list = np.array(
                        [self.objs[item].target.fitness for item in idx_candidates[0]]
                    )
                    idx_selected = idx_candidates[0][
                        np.argmin(fit_list)
                    ]  # Found the index of best fitness
                    pos_new[j] = self.objs[i].solution[j] + self.generator.uniform() * (
                            self.objs[i].solution[j] - self.objs[idx_selected].solution[j]
                    )
            if self.finished:
                break
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            # Step 4: Update herd immunity population
            if self.compare_target(
                    pop_new[idx].target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = pop_new[idx].copy()
            else:
                self.age_list[idx] += 1
            ## Calculate immunity mean of population
            fit_list = np.array([agent.target.fitness for agent in self.objs])
            delta_fx = np.mean(fit_list)
            if (
                    self.compare_fitness(
                        pop_new[idx].target.fitness, delta_fx, self.problem.minmax
                    )
                    and (self.immunity_type_list[idx] == 0)
                    and is_corona_list[idx]
            ):
                self.immunity_type_list[idx] = 1
                self.age_list[idx] = 1
            if self.compare_fitness(
                    delta_fx, pop_new[idx].target.fitness, self.problem.minmax
            ) and (self.immunity_type_list[idx] == 1):
                self.immunity_type_list[idx] = 2
                self.age_list[idx] = 0
            # Step 5: Fatality condition
            if (self.age_list[idx] >= self.max_age) and (
                    self.immunity_type_list[idx] == 1
            ):
                self.objs[idx] = self.generate_agent()
                self.immunity_type_list[idx] = 0
                self.age_list[idx] = 0
