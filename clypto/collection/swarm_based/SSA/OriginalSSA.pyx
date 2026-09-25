#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:22, 29/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.swarm_based.SSA.DevSSA cimport DevSSA
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalSSA(DevSSA):
    """
    The original version of: Sparrow Search Algorithm (SSA)

    Notes:
        + The paper contains some unclear equations and symbol
        + https://doi.org/10.1080/21642583.2019.1708830

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
        + PD (float): number of producers (percentage), default = 0.2
        + SD (float): number of sparrows who perceive the danger, default = 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SSA    >>> import numpy as np
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
    >>> model = SSA.OriginalSSA(epoch=1000, pop_size=50, ST = 0.8, PD = 0.2, SD = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Xue, J. and Shen, B., 2020. A novel swarm intelligence optimization approach:
    sparrow search algorithm. Systems Science & Control Engineering, 8(1), pp.22-34.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        ST: float = 0.8,
        PD: float = 0.2,
        SD: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ST (float): ST in [0.5, 1.0], safety threshold value, default = 0.8
            PD (float): number of producers (percentage), default = 0.2
            SD (float): number of sparrows who perceive the danger, default = 0.1
        """
        super().__init__(epoch, pop_size, ST, PD, SD, name=name, mode=mode)

    def evolve_agents(self, epoch):
        r2 = self.generator.uniform()  # R2 in [0, 1], the alarm value, random value
        pop_new = []
        for idx in range(0, self.pop_size):
            # Using equation (3) update the sparrow’s location;
            if idx < self.n1:
                if r2 < self.ST:
                    des = (idx + 1) / (
                            self.generator.uniform() * self.epoch + self.EPSILON
                    )
                    if des > 5:
                        des = self.generator.uniform()
                    x_new = self.objs[idx].solution * np.exp(des)
                else:
                    x_new = self.objs[idx].solution + self.generator.normal() * np.ones(
                        self.problem.n_dims
                    )
            else:
                # Using equation (4) update the sparrow’s location;
                _, x_p, worst = self.get_special_agents(
                    self.objs, n_best=1, n_worst=1, minmax=self.problem.minmax
                )
                g_best, g_worst = x_p[0], worst[0]
                if idx > int(self.pop_size / 2):
                    x_new = self.generator.normal() * np.exp(
                        (g_worst.solution - self.objs[idx].solution) / (idx + 1) ** 2
                    )
                else:
                    L = np.ones((1, self.problem.n_dims))
                    A = np.sign(self.generator.uniform(-1, 1, (1, self.problem.n_dims)))
                    A1 = A.T * np.linalg.inv(np.matmul(A, A.T)) * L
                    x_new = g_best.solution + np.matmul(
                        np.abs(self.objs[idx].solution - g_best.solution), A1
                    )
            pos_new = self.correct_solution(x_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    self.objs[idx], agent, self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
        self.objs, best, worst = self.get_special_agents(
            self.objs, n_best=1, n_worst=1, minmax=self.problem.minmax
        )
        g_best, g_worst = best[0], worst[0]
        pop2 = [agent.copy() for agent in self.objs[self.n2:]]
        child = []
        for idx in range(0, len(pop2)):
            #  Using equation (5) update the sparrow’s location;
            if self.compare_target(
                    self.objs[idx].target, g_best.target, self.problem.minmax
            ):
                x_new = pop2[idx].solution + self.generator.uniform(-1, 1) * (
                        np.abs(pop2[idx].solution - g_worst.solution)
                        / (pop2[idx].target.fitness - g_worst.target.fitness + self.EPSILON)
                )
            else:
                x_new = g_best.solution + self.generator.normal() * np.abs(
                    pop2[idx].solution - g_best.solution
                )
            pos_new = self.correct_solution(x_new)
            agent = self.generate_empty_agent(pos_new)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                pop2[idx] = self.get_better_agent(pop2[idx], agent, self.problem.minmax)
        if self.mode in self.AVAILABLE_MODES:
            child = self.update_target_for_population(child)
            pop2 = self.greedy_selection_population(pop2, child, self.problem.minmax)
        self.objs = self.objs[: self.n2] + pop2
