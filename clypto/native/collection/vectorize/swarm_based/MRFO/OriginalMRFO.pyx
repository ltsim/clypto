#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
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


cdef class OriginalMRFO(AgentListOptimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import MRFO    >>> import numpy as np
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
    >>> model = MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    cdef public object somersault_range

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        somersault_range: float = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "somersault_range"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.somersault_range = cy.validator(float, somersault_range, [1.0, 5.0], "somersault_range")

    def evolve_agents(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if self.generator.random() < 0.5:
                r1 = self.generator.uniform()
                beta = (
                        2
                        * np.exp(r1 * (self.epoch - epoch) / self.epoch)
                        * np.sin(2 * np.pi * r1)
                )

                if (epoch + 1) / self.epoch < self.generator.random():
                    x_rand = self.generator.uniform(self.problem.lb, self.problem.ub)
                    if idx == 0:
                        x_t1 = (
                                x_rand
                                + self.generator.uniform()
                                * (x_rand - self.objs[idx].solution)
                                + beta * (x_rand - self.objs[idx].solution)
                        )
                    else:
                        x_t1 = (
                                x_rand
                                + self.generator.uniform()
                                * (self.objs[idx - 1].solution - self.objs[idx].solution)
                                + beta * (x_rand - self.objs[idx].solution)
                        )
                else:
                    if idx == 0:
                        x_t1 = (
                                self.g_best.solution
                                + self.generator.uniform()
                                * (self.g_best.solution - self.objs[idx].solution)
                                + beta * (self.g_best.solution - self.objs[idx].solution)
                        )
                    else:
                        x_t1 = (
                                self.g_best.solution
                                + self.generator.uniform()
                                * (self.objs[idx - 1].solution - self.objs[idx].solution)
                                + beta * (self.g_best.solution - self.objs[idx].solution)
                        )
            # Chain foraging (Eq. 1,2)
            else:
                r = self.generator.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = (
                            self.objs[idx].solution
                            + r * (self.g_best.solution - self.objs[idx].solution)
                            + alpha * (self.g_best.solution - self.objs[idx].solution)
                    )
                else:
                    x_t1 = (
                            self.objs[idx].solution
                            + r * (self.objs[idx - 1].solution - self.objs[idx].solution)
                            + alpha * (self.g_best.solution - self.objs[idx].solution)
                    )
            pos_new = self.correct_solution(x_t1)
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
        _, g_best = self.update_global_best_agent(self.objs, save=False)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = self.objs[idx].solution + self.somersault_range * (
                    self.generator.uniform() * g_best.solution
                    - self.generator.uniform() * self.objs[idx].solution
            )
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.objs[idx] = self.get_better_agent(
                    self.objs[idx], agent, self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.objs = self.greedy_selection_population(
                self.objs, pop_child, self.problem.minmax
            )
