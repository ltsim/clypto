#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:01, 16/08/2025 ----------%
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


cdef class OriginalFDO(AgentListOptimizer):
    """
    The original version of: Fitness Dependent Optimizer (FDO)

    Notes:
        + https://doi.org/10.1109/ACCESS.2019.2907012
        + Inspired by the bee swarming reproductive process, this algorithm optimizes solutions based on their fitness values.
        + This algorithm mainly relies on Lévy flight techniques. Thanks to this method of generating random numbers
        according to the Lévy distribution, it is able to converge. However, in the design of the fitness weight
        condition, it is almost impossible for an update to occur when the fitness weight equals 1. This is the main drawback.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import FDO    >>> import numpy as np
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
    >>> model = FDO.OriginalFDO(epoch=1000, pop_size=50, weight_factor=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdullah, J. M., & Ahmed, T. (2019).
    Fitness dependent optimizer: inspired by the bee swarming reproductive process. IEEe Access, 7, 43473-43486.
    """

    cdef public object weight_factor
    cdef public object pop_pace

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        weight_factor = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            weight_factor (float): factor to adjust the fitness weight calculation, default = 0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "weight_factor"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.weight_factor = cy.validator(float, weight_factor, [0.0, 1.0], "weight_factor")

    cdef void before_main_loop(self):
        self.pop_pace = [
                            0,
                        ] * self.pop_size

    def get_fit_weight(self, best_fit, current_fit, weight_factor=0.1):
        if best_fit == 0:
            return 0
        else:
            if self.problem.minmax == "min":
                if best_fit < (0.05 * current_fit):
                    return 0.2
                else:
                    return best_fit / current_fit - weight_factor
            else:
                if best_fit > (0.05 * current_fit):
                    return 0.2
                else:
                    return weight_factor - best_fit / current_fit

    def get_into_levy_bound(self, pos_new):
        levy = self.get_levy_flight_step(
            beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1
        )
        levy_up = self.problem.ub * np.abs(levy)
        levy_lb = self.problem.lb * np.abs(levy)
        pos_new = np.select(
            [pos_new > self.problem.ub, pos_new < self.problem.lb],
            [levy_up, levy_lb],
            default=pos_new,
        )
        return pos_new

    def evolve_agents(self, epoch):
        # Update positions for each thief
        for idx in range(self.pop_size):
            fw = self.get_fit_weight(
                self.g_best.target.fitness,
                self.objs[idx].target.fitness,
                self.weight_factor,
            )
            dist = self.g_best.solution - self.objs[idx].solution
            levy = self.get_levy_flight_step(
                beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1
            )
            if fw == 1:
                pace = self.objs[idx].solution * levy
            elif fw == 0:
                pace = dist * levy
            else:
                pace = dist * fw * np.sign(levy)
            self.pop_pace[idx] = pace
            pos_new = self.objs[idx].solution + pace
            pos_new = self.get_into_levy_bound(pos_new)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            # Check if new position is better
            if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx] = agent
            else:
                # Alternative update strategy
                dist = self.g_best.solution - pos_new
                pos_new = pos_new + (dist * fw) + self.pop_pace[idx]
                pos_new = self.get_into_levy_bound(pos_new)
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(
                        agent.target, self.objs[idx].target, self.problem.minmax
                ):
                    self.objs[idx] = agent
                else:
                    # Third update strategy
                    levy = self.get_levy_flight_step(
                        beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1
                    )
                    pos_new = self.objs[idx].solution + self.objs[idx].solution * levy
                    pos_new = self.get_into_levy_bound(pos_new)
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(
                            agent.target, self.objs[idx].target, self.problem.minmax
                    ):
                        self.objs[idx] = agent
