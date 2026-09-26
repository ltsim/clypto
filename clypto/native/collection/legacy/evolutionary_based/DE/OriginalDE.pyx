#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class OriginalDE(_LegacyOptimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + wf (float): [-1., 1.0], weighting factor, default = 0.1
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/current-to-rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.OriginalDE(epoch=1000, pop_size=50, wf = 0.7, cr = 0.9, strategy = 0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        wf: float = 0.1,
        cr: float = 0.9,
        strategy: int = 0,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.1
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.wf = self.validator.check_float("wf", wf, (-3.0, 3.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.strategy = self.validator.check_int("strategy", strategy, [0, 5])
        self.set_parameters(["epoch", "pop_size", "wf", "cr", "strategy"])
        self.sort_flag = False

    def mutation__(self, current_pos, new_pos):
        condition = self.generator.random(self.problem.n_dims) < self.cr
        pos_new = np.where(condition, new_pos, current_pos)
        return self.correct_solution(pos_new)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        if self.strategy == 0:
            # Choose 3 random element and different to i
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 3, replace=False
                )
                pos_new = self.pop[idx_list[0]].solution + self.wf * (
                    self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution
                )
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(
                        agent, self.pop[idx], self.problem.minmax
                    )
        elif self.strategy == 1:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                )
                pos_new = self.g_best.solution + self.wf * (
                    self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution
                )
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(
                        agent, self.pop[idx], self.problem.minmax
                    )
        elif self.strategy == 2:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 4, replace=False
                )
                pos_new = (
                    self.g_best.solution
                    + self.wf
                    * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution)
                    + self.wf
                    * (self.pop[idx_list[2]].solution - self.pop[idx_list[3]].solution)
                )
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(
                        agent, self.pop[idx], self.problem.minmax
                    )
        elif self.strategy == 3:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 5, replace=False
                )
                pos_new = (
                    self.pop[idx_list[0]].solution
                    + self.wf
                    * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution)
                    + self.wf
                    * (self.pop[idx_list[3]].solution - self.pop[idx_list[4]].solution)
                )
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(
                        agent, self.pop[idx], self.problem.minmax
                    )
        elif self.strategy == 4:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                )
                pos_new = (
                    self.pop[idx].solution
                    + self.wf * (self.g_best.solution - self.pop[idx].solution)
                    + self.wf
                    * (self.pop[idx_list[0]].solution - self.pop[idx_list[1]].solution)
                )
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(
                        agent, self.pop[idx], self.problem.minmax
                    )
        else:
            for idx in range(0, self.pop_size):
                idx_list = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx}), 3, replace=False
                )
                pos_new = (
                    self.pop[idx].solution
                    + self.wf
                    * (self.pop[idx_list[0]].solution - self.pop[idx].solution)
                    + self.wf
                    * (self.pop[idx_list[1]].solution - self.pop[idx_list[2]].solution)
                )
                pos_new = self.mutation__(self.pop[idx].solution, pos_new)
                agent = self.generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(pos_new)
                    self.pop[idx] = self.get_better_agent(
                        agent, self.pop[idx], self.problem.minmax
                    )
        if self.mode in self.AVAILABLE_MODES:
            pop = self.update_target_for_population(pop)
            self.pop = self.greedy_selection_population(
                self.pop, pop, self.problem.minmax
            )
