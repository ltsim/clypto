#!/usr/bin/env python
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

from math import gamma
import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class _ModifiedSLOAgent(LegacyAgent):
    cdef public object local_solution
    cdef public object local_target


cdef class ModifiedSLO(LegacyOptimizer):
    """
    The original version of: Modified Sea Lion Optimization (M-SLO)

    Notes:
        + Local best idea in PSO is inspired
        + Levy-flight technique is used
        + Shrink encircling idea is used

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import SLO    >>> import numpy as np
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
    >>> model = SLO.ModifiedSLO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
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

    def _generate_empty_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        local_pos = self.problem.bounds.low + self.problem.bounds.up - solution
        local_pos = self._correct_solution(local_pos)
        return _ModifiedSLOAgent(solution=solution, local_solution=local_pos)

    def _generate_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        agent = self._generate_empty_agent(solution)
        target = self._get_target(agent.solution)
        local_target = self._get_target(agent.local_solution)
        if self._compare_target(target, local_target, self.problem.sense):
            t1 = agent.local_solution.copy()
            t2 = agent.solution.copy()
            agent.update(
                solution=t1, target=local_target, local_solution=t2, local_target=target
            )
        else:
            t1 = agent.solution.copy()
            t2 = agent.local_solution.copy()
            agent.update(
                solution=t1, target=target, local_solution=t2, local_target=local_target
            )
        return agent

    def shrink_encircling_levy__(self, current_pos, epoch, dist, c, beta=1):
        up = gamma(1 + beta) * np.sin(np.pi * beta / 2)
        down = gamma((1.0 + beta) / 2.0) * beta * np.power(2.0, (beta - 1.0) / 2.0)
        xich_ma_1 = np.power(up / down, 1 / beta)
        xich_ma_2 = 1.0
        a = self.generator.normal(0, xich_ma_1, 1)
        b = self.generator.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (np.power(np.abs(b), 1 / beta)) * dist * c
        D = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        levy = LB * D
        return (
            current_pos - np.sqrt(epoch + 1) * np.sign(self.generator.random() - 0.5)
        ) * levy

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """

        c = 2.0 - 2.0 * epoch / self.epoch
        if c > 1:
            pa = 0.3  # At the beginning of the process, the probability for shrinking encircling is small
        else:
            pa = 0.7  # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey
        SP_leader = self.generator.uniform(0, 1)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            if SP_leader >= 0.6:
                pos_new = (
                    np.cos(2 * np.pi * self.generator.normal(0, 1))
                    * np.abs(self.g_best.solution - self.pop[idx].solution)
                    + self.g_best.solution
                )
            else:
                if self.generator.uniform() < pa:
                    dist1 = self.generator.uniform() * np.abs(
                        2 * self.g_best.solution - self.pop[idx].solution
                    )
                    pos_new = self.shrink_encircling_levy__(
                        self.pop[idx].solution, epoch, dist1, c
                    )
                else:
                    rand_SL = self.pop[
                        self.generator.integers(0, self.pop_size)
                    ].local_solution
                    rand_SL = 2 * self.g_best.solution - rand_SL
                    pos_new = rand_SL - c * np.abs(
                        self.generator.uniform() * rand_SL - self.pop[idx].solution
                    )
            pos_new = self._correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self._get_target(agent.solution)
        pop_new = self._update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            if self._compare_target(
                pop_new[idx].target, self.pop[idx].target, self.problem.sense
            ):
                self.pop[idx] = pop_new[idx].copy()
                if self._compare_target(
                    pop_new[idx].target, self.pop[idx].local_target, self.problem.sense
                ):
                    self.pop[idx].local_solution = pop_new[idx].solution.copy()
                    self.pop[idx].local_target = pop_new[idx].target.copy()
