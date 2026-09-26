#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class WMQIMRFO(AgentListOptimizer):
    """
    The original version of: Wavelet Mutation and Quadratic Interpolation MRFO (WMQIMRFO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2021.108071

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2
        + pm (float): (0.0, 1.0), probability mutation, default = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import MRFO    >>> import numpy as np
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
    >>> model = MRFO.WMQIMRFO(epoch=1000, pop_size=50, somersault_range = 2.0, pm=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] G. Hu, M. Li, X. Wang et al., An enhanced manta ray foraging optimization algorithm for shape optimization of
    complex CCG-Ball curves, Knowledge-Based Systems (2022), doi: https://doi.org/10.1016/j.knosys.2021.108071.
    """

    cdef public object somersault_range
    cdef public object pm

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        somersault_range: float = 2.0,
        pm: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
            pm (float): probability mutation, default = 0.5
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "somersault_range", "pm"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.somersault_range = cy.validator(float, somersault_range, [1.0, 5.0], "somersault_range")
        self.pm = cy.validator(float, pm, (0.0, 1.0), "pm")

    def _evolve_agents(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            x_t = self.objs[idx].solution
            x_t1 = self.objs[idx - 1].solution

            ## Morlet wavelet mutation strategy
            ## Goal is to jump out of local optimum --> Performed in exploration stage
            s_constant = 2.0
            a = s_constant * (1.0 / s_constant) ** (1.0 - epoch / self.epoch)
            theta = self.generator.uniform(-2.5 * a, 2.5 * a)
            x = theta / a
            w = np.exp(-(x ** 2) / 2) * np.cos(5 * x)
            xichma = 1.0 / np.sqrt(a) * w

            if self.generator.random() < 0.5:  # Control parameter adjustment
                coef = np.log(1 + (np.e - 1.0) * epoch / self.epoch)  # Eq. 3.11

                r1 = self.generator.uniform()
                beta = (
                        2
                        * np.exp(r1 * (self.epoch - epoch) / self.epoch)
                        * np.sin(2 * np.pi * r1)
                )

                if coef < self.generator.random():  # Cyclone foraging
                    x_rand = self.problem.generate_solution()
                    if self.generator.random() < self.pm:  # Morlet wavelet mutation
                        if idx == 0:
                            pos_new = (
                                    x_rand
                                    + self.generator.random() * (x_rand - x_t)
                                    + beta * (x_rand - x_t)
                            )
                        else:
                            pos_new = (
                                    x_rand
                                    + self.generator.random() * (x_t1 - x_t)
                                    + beta * (x_rand - x_t)
                            )
                    else:
                        conditions = (
                                self.generator.uniform(0, 1, self.problem.n_dims) > 0.5
                        )
                        if idx == 0:
                            t1 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_rand - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (self.problem.bounds.up - x_t)
                            )
                            t2 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_rand - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (x_t - self.problem.bounds.low)
                            )
                        else:
                            t1 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_t1 - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (self.problem.bounds.up - x_t)
                            )
                            t2 = (
                                    x_rand
                                    + self.generator.random(self.problem.n_dims)
                                    * (x_t1 - x_t)
                                    + beta * (x_rand - x_t)
                                    + xichma * (x_t - self.problem.bounds.low)
                            )
                        pos_new = np.where(conditions, t1, t2)
                else:
                    if idx == 0:
                        pos_new = (
                                self.g_best.solution
                                + self.generator.random() * (self.g_best.solution - x_t)
                                + beta * (self.g_best.solution - x_t)
                        )
                    else:
                        pos_new = (
                                self.g_best.solution
                                + self.generator.random() * (x_t1 - x_t)
                                + beta * (self.g_best.solution - x_t)
                        )
            else:  # Chain foraging (Eq. 1,2)
                r = self.generator.random()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    pos_new = (
                            x_t
                            + r * (self.g_best.solution - x_t)
                            + alpha * (self.g_best.solution - x_t)
                    )
                else:
                    pos_new = (
                            x_t + r * (x_t1 - x_t) + alpha * (self.g_best.solution - x_t)
                    )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.objs[idx] = self._get_better_agent(
                    self.objs[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.objs = self._greedy_selection_population(
                self.objs, pop_new, self.problem.sense
            )
        _, g_best = self._update_global_best_agent(self.objs)

        # Somersault foraging   (Eq. 8)
        pop_child = []
        for idx in range(0, self.pop_size):
            pos_new = self.objs[idx].solution + self.somersault_range * (
                    self.generator.random() * g_best.solution
                    - self.generator.random() * self.objs[idx].solution
            )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.objs[idx] = self._get_better_agent(
                    self.objs[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self._update_target_for_population(pop_child)
            self.objs = self._greedy_selection_population(
                self.objs, pop_child, self.problem.sense
            )
        self.objs, g_best = self._update_global_best_agent(self.objs)

        # Quadratic Interpolation
        pop_new = []
        for idx in range(0, self.pop_size):
            idx2, idx3 = idx + 1, idx + 2
            if idx == self.pop_size - 2:
                idx2, idx3 = idx + 1, 0
            if idx == self.pop_size - 1:
                idx2, idx3 = 0, 1
            f1, f2, f3 = (
                self.objs[idx].target.fitness,
                self.objs[idx2].target.fitness,
                self.objs[idx3].target.fitness,
            )
            x1, x2, x3 = (
                self.objs[idx].solution,
                self.objs[idx2].solution,
                self.objs[idx3].solution,
            )
            a = (
                    f1 / ((x1 - x2) * (x1 - x3) + self.EPSILON)
                    + f2 / ((x2 - x1) * (x2 - x3) + self.EPSILON)
                    + f3 / ((x3 - x1) * (x3 - x2) + self.EPSILON)
            )
            gx = (
                         (x3 ** 2 - x2 ** 2) * f1 + (x1 ** 2 - x3 ** 2) * f2 + (x2 ** 2 - x1 ** 2) * f3
                 ) / (2 * ((x3 - x2) * f1 + (x1 - x3) * f2 + (x2 - x1) * f3) + self.EPSILON)
            pos_new = np.where(a > 0, gx, x1)
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.objs[idx] = self._get_better_agent(
                    self.objs[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.objs = self._greedy_selection_population(
                self.objs, pop_new, self.problem.sense
            )
