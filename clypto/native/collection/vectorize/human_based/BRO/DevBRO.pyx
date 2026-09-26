#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 09:17, 09/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from scipy.spatial.distance import cdist

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class DevBRO(AgentListOptimizer):
    """
    The developed version: Battle Royale Optimization (BRO)

    Notes:
        + The flow of algorithm is changed. Thrid loop is removed

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import BRO    >>> import numpy as np
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
    >>> model = BRO.DevBRO(epoch=1000, pop_size=50, threshold = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        threshold: float = 3,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "threshold"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.threshold = cy.validator(float, threshold, [1, 10], "threshold")

    cdef void initialize_variables(self):
        shrink = np.ceil(np.log10(self.epoch))
        self.dyn_delta = np.round(self.epoch / shrink)
        self.lb_updated = self.problem.lb.copy()
        self.ub_updated = self.problem.ub.copy()

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        damage = 0
        return FieldAgent(solution=solution, damage=damage)

    def get_idx_min__(self, data):
        k_zero = np.count_nonzero(data == 0)
        if k_zero == len(data):
            return self.generator.choice(range(0, k_zero))
        ## 1st: Partition sorting, not good solution here.
        # return np.argpartition(data, k_zero)[k_zero]
        ## 2nd: Faster
        return np.where(data == np.min(data[data != 0]))[0][0]

    def find_idx_min_distance__(self, target_pos=None, pop=None):
        list_pos = np.array([pop[idx].solution for idx in range(0, self.pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, "euclidean")
        dist_list = np.reshape(dist_list, (-1))
        return self.get_idx_min__(dist_list)

    def evolve_agents(self, epoch):
        for idx in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            jdx = self.find_idx_min_distance__(self.objs[idx].solution, self.objs)
            if self.compare_target(
                self.objs[idx].target, self.objs[jdx].target, self.problem.minmax
            ):
                ## Update Winner based on global best solution
                pos_new = self.objs[idx].solution + self.generator.normal(
                    0, 1
                ) * np.mean(
                    np.array([self.objs[idx].solution, self.g_best.solution]), axis=0
                )
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                dam_new = (
                    self.objs[idx].damage - 1
                )  ## Substract damaged hurt -1 to go next battle
                agent.damage = dam_new
                self.objs[idx] = agent
                ## Update Loser
                if (
                    self.objs[jdx].damage < self.threshold
                ):  ## If loser not dead yet, move it based on general
                    pos_new = self.generator.uniform() * (
                        np.maximum(self.objs[jdx].solution, self.g_best.solution)
                        - np.minimum(self.objs[jdx].solution, self.g_best.solution)
                    ) + np.maximum(self.objs[jdx].solution, self.g_best.solution)
                    dam_new = self.objs[jdx].damage + 1
                    self.objs[jdx].target = self.get_target(self.objs[jdx].solution)
                else:  ## Loser dead and respawn again
                    pos_new = self.generator.uniform(
                        self.lb_updated, self.ub_updated
                    )
                    dam_new = 0
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                agent.damage = dam_new
                self.objs[jdx] = agent
            else:
                ## Update Loser by following position of Winner
                self.objs[idx] = self.objs[jdx].copy()
                ## Update Winner by following position of General to protect the King and General
                pos_new = self.objs[jdx].solution + self.generator.uniform() * (
                    self.g_best.solution - self.objs[jdx].solution
                )
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                agent.damage = 0
                self.objs[jdx] = agent
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array(
                [self.objs[idx].solution for idx in range(0, self.pop_size)]
            )
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best.solution - pos_std
            ub = self.g_best.solution + pos_std
            self.lb_updated = np.clip(
                lb, self.lb_updated, self.ub_updated
            )
            self.ub_updated = np.clip(
                ub, self.lb_updated, self.ub_updated
            )
            self.dyn_delta += np.round(self.dyn_delta / 2)
