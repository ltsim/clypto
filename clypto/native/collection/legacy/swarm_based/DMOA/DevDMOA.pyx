#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class DevDMOA(_LegacyOptimizer):
    """
    The developed version of: Dwarf Mongoose Optimization Algorithm (DMOA)

    Notes:
        1. Removed the parameter n_baby_sitter
        2. Changed in section # Next Mongoose position
        3. Removed the meaningless variable tau

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import DMOA    >>> import numpy as np
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
    >>> model = DMOA.DevDMOA(epoch=1000, pop_size=50, peep = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, peep: float = 2, **kwargs: object
    ) -> None:
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.peep = self.validator.check_float("peep", peep, [1, 10.0])
        self.set_parameters(["epoch", "pop_size", "peep"])
        self.is_parallelizable = False
        self.sort_flag = False

    def initialize_variables(self):
        self.C = np.zeros(self.pop_size)
        self.L = np.round(0.6 * self.epoch)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Abandonment Counter
        CF = (1.0 - epoch / self.epoch) ** (2.0 * epoch / self.epoch)
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        mean_cost = np.mean(fit_list)
        fi = np.exp(-fit_list / mean_cost)

        ## Foraging led by Alpha female
        for idx in range(0, self.pop_size):
            alpha = self.get_index_roulette_wheel_selection(fi)
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, alpha}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[alpha].solution + phi * (
                    self.pop[alpha].solution - self.pop[k].solution
            )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(
                    agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        ## Scout group
        SM = np.zeros(self.pop_size)
        for idx in range(0, self.pop_size):
            k = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            ## Define Vocalization Coeff.
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            new_pos = self.pop[idx].solution + phi * (
                    self.pop[idx].solution - self.pop[k].solution
            )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            ## Sleeping mould
            SM[idx] = (agent.target.fitness - self.pop[idx].target.fitness) / (
                    np.max([agent.target.fitness, self.pop[idx].target.fitness])
                    + self.EPSILON
            )
            if self.compare_target(
                    agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
            else:
                self.C[idx] += 1

        ## Baby sitters
        for idx in range(0, self.pop_size):
            if self.C[idx] >= self.L:
                self.pop[idx] = self.generate_agent()
                self.C[idx] = 0

        ## Next Mongoose position
        new_tau = np.mean(SM)
        for idx in range(0, self.pop_size):
            phi = (self.peep / 2) * self.generator.uniform(-1, 1, self.problem.n_dims)
            if new_tau > SM[idx]:
                new_pos = self.g_best.solution - CF * phi * (
                        self.g_best.solution - SM[idx] * self.pop[idx].solution
                )
            else:
                new_pos = self.pop[idx].solution + CF * phi * (
                        self.g_best.solution - SM[idx] * self.pop[idx].solution
                )
            new_pos = self.correct_solution(new_pos)
            agent = self.generate_agent(new_pos)
            if self.compare_target(
                    agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
