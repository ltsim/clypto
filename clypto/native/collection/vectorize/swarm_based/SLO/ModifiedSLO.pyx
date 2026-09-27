#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 15:05, 03/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from math import gamma
import numpy as np



cimport clypto.core as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class ModifiedSLO(AgentListOptimizer):
    """
    The original version of: Modified Sea Lion Optimization (M-SLO)

    Notes:
        + Local best idea in PSO is inspired
        + Levy-flight technique is used
        + Shrink encircling idea is used

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import SLO    >>> import numpy as np
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
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
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
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        local_pos = self.problem.bounds.low + self.problem.bounds.up - solution
        local_pos = self.correct_solution(local_pos)
        return FieldAgent(solution=solution, local_solution=local_pos)

    def generate_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        agent = self.create_agent(solution)
        target = self.evaluate_solution(agent.solution)
        local_best = self.evaluate_solution(agent.local_solution)
        if cy.is_better(target, local_best, self.problem.sense):
            t1 = agent.local_solution.copy()
            t2 = agent.solution.copy()
            agent.update_solution(local_best, t1)
            agent.update(local_solution=t2, local_best=target)
        else:
            t1 = agent.solution.copy()
            t2 = agent.local_solution.copy()
            agent.update_solution(target, t1)
            agent.update(local_solution=t2, local_best=local_best)
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

    def evolve_agents(self, epoch):

        c = 2.0 - 2.0 * epoch / self.epoch
        if c > 1:
            pa = 0.3  # At the beginning of the process, the probability for shrinking encircling is small
        else:
            pa = 0.7  # But at the end of the process, it become larger. Because sea lion are shrinking encircling prey
        SP_leader = self.generator.uniform(0, 1)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.objs[idx].copy()
            if SP_leader >= 0.6:
                pos_new = (
                    np.cos(2 * np.pi * self.generator.normal(0, 1))
                    * np.abs(self.g_best.solution - self.objs[idx].solution)
                    + self.g_best.solution
                )
            else:
                if self.generator.uniform() < pa:
                    dist1 = self.generator.uniform() * np.abs(
                        2 * self.g_best.solution - self.objs[idx].solution
                    )
                    pos_new = self.shrink_encircling_levy__(
                        self.objs[idx].solution, epoch, dist1, c
                    )
                else:
                    rand_SL = self.objs[
                        self.generator.integers(0, self.pop_size)
                    ].local_solution
                    rand_SL = 2 * self.g_best.solution - rand_SL
                    pos_new = rand_SL - c * np.abs(
                        self.generator.uniform() * rand_SL - self.objs[idx].solution
                    )
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        pop_new = self.evaluate_agents(pop_new)
        for idx in range(0, self.pop_size):
            if cy.is_better(pop_new[idx], self.objs[idx], self.problem.sense):
                self.objs[idx] = pop_new[idx].copy()
                if cy.is_better(pop_new[idx], self.objs[idx].local_best, self.problem.sense):
                    self.objs[idx].local_solution = pop_new[idx].solution.copy()
                    self.objs[idx].local_best = pop_new[idx].copy()
