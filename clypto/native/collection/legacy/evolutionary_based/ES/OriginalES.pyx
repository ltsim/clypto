#!/usr/bin/env python
# Created by "Thieu" at 18:14, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent
from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class _OriginalESAgent(_LegacyAgent):
    cdef public object strategy
    def __init__(self, solution=None, target=None, strategy=None):
        _LegacyAgent.__init__(self, solution, target)
        self.strategy = strategy
    cpdef object copy(self):
        return _OriginalESAgent(
            self.solution, None if self.target is None else self.target.copy(),
            self.strategy,
        )
    def update(self, **kwargs):
        if "strategy" in kwargs:
            self.strategy = kwargs.pop("strategy")
        _LegacyAgent.update(self, **kwargs)


cdef class OriginalES(_LegacyOptimizer):
    """
    The original version of: Evolution Strategies (ES)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + lamda (float): [0.5, 1.0], Percentage of child agents evolving in the next generation

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import ES    >>> import numpy as np
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
    >>> model = ES.OriginalES(epoch=1000, pop_size=50, lamda = 0.75)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        lamda: float = 0.75,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            lamda (float): Percentage of child agents evolving in the next generation, default=0.75
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.lamda = self.validator.check_float("lamda", lamda, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "lamda"])
        self.n_child = int(self.lamda * self.pop_size)
        self.sort_flag = True

    def initialize_variables(self):
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance)
        return _OriginalESAgent(solution=solution, strategy=strategy)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.n_child):
            pos_new = self.pop[idx].solution + self.pop[
                idx
            ].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            tau = np.sqrt(2.0 * self.problem.n_dims) ** (-1.0)
            tau_p = np.sqrt(2.0 * np.sqrt(self.problem.n_dims)) ** (-1.0)
            strategy = np.exp(
                tau_p * self.generator.normal(0, 1.0, self.problem.n_dims)
                + tau * self.generator.normal(0, 1.0, self.problem.n_dims)
            )
            agent = self.generate_empty_agent(pos_new)
            agent.update(solution=pos_new, strategy=strategy)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        self.pop = self.get_sorted_and_trimmed_population(
            child + self.pop, self.pop_size, self.problem.minmax
        )
