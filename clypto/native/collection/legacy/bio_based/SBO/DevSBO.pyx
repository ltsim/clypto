#!/usr/bin/env python
# Created by "Thieu" at 12:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class DevSBO(LegacyOptimizer):
    """
    The developed version: Satin Bowerbird Optimizer (SBO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2017.01.006

    Notes:
        The original version can't handle negative fitness value.
        I remove all third loop for faster training, remove equation (1, 2) in the paper, calculate probability by roulette-wheel.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.5, 3.0] -> better [0.5, 2.0], the greatest step size
        + p_m (float): (0, 1.0) -> better [0.01, 0.2], mutation probability
        + psw (float): (0, 1.0) -> better [0.01, 0.1], proportion of space width (z in the paper)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import SBO    >>> import numpy as np
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
    >>> model = SBO.DevSBO(epoch=1000, pop_size=50, alpha = 0.9, p_m =0.05, psw = 0.02)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            alpha: float = 0.94,
            p_m: float = 0.05,
            psw: float = 0.02,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the greatest step size, default=0.94
            p_m (float): mutation probability, default=0.05
            psw (float): proportion of space width (z in the paper), default=0.02
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.alpha = self.validator.check_float("alpha", alpha, [0.5, 3.0])
        self.p_m = self.validator.check_float("p_m", p_m, (0, 1.0))
        self.psw = self.validator.check_float("psw", psw, (0, 1.0))
        self._set_parameters(["epoch", "pop_size", "alpha", "p_m", "psw"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # (percent of the difference between the upper and lower limit (Eq. 7))
        self.sigma = self.psw * (self.problem.bounds.up - self.problem.bounds.low)

        ## Calculate the probability of bowers using my equation
        fit_list = np.array([agent.target.fitness for agent in self.pop])
        pop_new = []
        for idx in range(0, self.pop_size):
            ### Select a bower using roulette wheel
            rdx = self._get_index_roulette_wheel_selection(fit_list)
            ### Calculating Step Size
            lamda = self.alpha * self.generator.uniform()
            pos_new = self.pop[idx].solution + lamda * (
                    (self.pop[rdx].solution + self.g_best.solution) / 2
                    - self.pop[idx].solution
            )
            ### Mutation
            temp = (
                    self.pop[idx].solution
                    + self.generator.normal(0, 1, self.problem.n_dims) * self.sigma
            )
            pos_new = np.where(
                self.generator.random(self.problem.n_dims) < self.p_m, temp, pos_new
            )
            ### In-bound position
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    agent, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )
