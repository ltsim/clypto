#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalESOA(AgentListOptimizer):
    """
    The original version of: Egret Swarm Optimization Algorithm (ESOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/115595-egret-swarm-optimization-algorithm-esoa
        2. https://www.mdpi.com/2313-7673/7/4/144

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import ESOA    >>> import numpy as np
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
    >>> model = ESOA.OriginalESOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, Z., Francis, A., Li, S., Liao, B., Xiao, D., Ha, T. T., ... & Cao, X. (2022). Egret Swarm Optimization Algorithm:
    An Evolutionary Computation Approach for Model Free Optimization. Biomimetics, 7(4), 144.
    """

    cdef public object beta1
    cdef public object beta2

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        weights = self.generator.uniform(-1.0, 1.0, self.problem.n_dims)
        m = np.zeros(self.problem.n_dims)
        v = np.zeros(self.problem.n_dims)
        return FieldAgent(
            solution=solution, weights=weights, local_solution=solution.copy(), m=m, v=v
        )

    def generate_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        agent.g = (
            np.sum(agent.weights * agent.solution) - agent.target.fitness
        ) * agent.solution
        return agent

    cdef void initialize_variables(self):
        self.beta1 = 0.9
        self.beta2 = 0.99

    def evolve_agents(self, epoch):
        hop = self.problem.ub - self.problem.lb
        for idx in range(0, self.pop_size):
            # Individual Direction
            p_d = self.objs[idx].local_solution - self.objs[idx].solution
            p_d = p_d * (
                self.objs[idx].local_target.fitness - self.objs[idx].target.fitness
            )
            p_d = p_d / (np.sum(p_d) ** 2 + self.EPSILON)
            d_p = p_d + self.objs[idx].g

            # Group Direction
            c_d = self.g_best.solution - self.objs[idx].solution
            c_d = c_d * (self.g_best.target.fitness - self.objs[idx].target.fitness)
            c_d = c_d / (np.sum(c_d) ** 2 + self.EPSILON)
            d_g = c_d + self.g_best.g

            # Gradient Estimation
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            g = (1 - r1 - r2) * self.objs[idx].g + r1 * d_p + r2 * d_g
            g = g / (np.sum(g) + self.EPSILON)

            self.objs[idx].m = self.beta1 * self.objs[idx].m + (1 - self.beta1) * g
            self.objs[idx].v = self.beta2 * self.objs[idx].v + (1 - self.beta2) * g**2
            self.objs[idx].weights -= self.objs[idx].m / (
                np.sqrt(self.objs[idx].v) + self.EPSILON
            )

            # Advice Forward
            x_0 = (
                self.objs[idx].solution
                + np.exp(-1.0 / (0.1 * self.epoch)) * 0.1 * hop * g
            )
            x_0 = self.correct_solution(x_0)
            y_0 = self.get_target(x_0)

            # Random Search
            r3 = self.generator.uniform(-np.pi / 2, np.pi / 2, self.problem.n_dims)
            x_n = self.objs[idx].solution + np.tan(r3) * hop / epoch * 0.5
            x_n = self.correct_solution(x_n)
            y_n = self.get_target(x_n)

            # Encircling Mechanism
            d = self.objs[idx].local_solution - self.objs[idx].solution
            d_g = self.g_best.solution - self.objs[idx].solution
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            x_m = (1 - r1 - r2) * self.objs[idx].solution + r1 * d + r2 * d_g
            x_m = self.correct_solution(x_m)
            y_m = self.get_target(x_m)

            # Discriminant Condition
            y_list_compare = [y_0.fitness, y_n.fitness, y_m.fitness]
            y_list = [y_0, y_n, y_m]
            x_list = [x_0, x_n, x_m]
            if self.problem.minmax == "min":
                id_best = np.argmin(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]
            else:
                id_best = np.argmax(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]

            if self.compare_target(y_best, self.objs[idx].target, self.problem.minmax):
                self.objs[idx].solution = x_best
                self.objs[idx].target = y_best
                if self.compare_target(
                    y_best, self.objs[idx].local_target, self.problem.minmax
                ):
                    self.objs[idx].local_solution = x_best
                    self.objs[idx].local_target = y_best
                    self.objs[idx].g = (
                        np.sum(self.objs[idx].weights * self.objs[idx].solution)
                        - self.objs[idx].target.fitness
                    ) * self.objs[idx].solution
            else:
                if self.generator.random() < 0.3:
                    self.objs[idx].solution = x_best
                    self.objs[idx].target = y_best
