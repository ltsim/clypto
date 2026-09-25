#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class OriginalNMRA(_LegacyOptimizer):
    """
    The original version of: Naked Mole-Rat Algorithm (NMRA)

    Links:
        1. https://www.doi.org10.1007/s00521-019-04464-7

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import NMRA    >>> import numpy as np
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
    >>> model = NMRA.OriginalNMRA(epoch=1000, pop_size=50, pb = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Salgotra, R. and Singh, U., 2019. The naked mole-rat algorithm. Neural Computing and Applications, 31(12), pp.8837-8857.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pb: float = 0.75,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): probability of breeding, default = 0.75
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.pb = self.validator.check_float("pb", pb, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "pb"])
        self.sort_flag = True
        self.size_b = int(self.pop_size / 5)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution.copy()
            if idx < self.size_b:  # breeding operators
                if self.generator.uniform() < self.pb:
                    alpha = self.generator.uniform()
                    pos_new = (1 - alpha) * self.pop[idx].solution + alpha * (
                            self.g_best.solution - self.pop[idx].solution
                    )
            else:  # working operators
                t1, t2 = self.generator.choice(
                    range(self.size_b, self.pop_size), 2, replace=False
                )
                pos_new = self.pop[idx].solution + self.generator.uniform() * (
                        self.pop[t1].solution - self.pop[t2].solution
                )
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    self.pop[idx], agent, self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop = self.greedy_selection_population(
                self.pop, pop_new, self.problem.minmax
            )
