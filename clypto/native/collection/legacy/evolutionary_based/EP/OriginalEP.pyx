#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent
from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class _OriginalEPAgent(_LegacyAgent):
    cdef public object strategy
    cdef public object win
    def __init__(self, solution=None, target=None, strategy=None, win=None):
        _LegacyAgent.__init__(self, solution, target)
        self.strategy = strategy
        self.win = win
    cpdef object copy(self):
        return _OriginalEPAgent(
            self.solution, None if self.target is None else self.target.copy(),
            self.strategy,
            self.win,
        )
    def update(self, **kwargs):
        if "strategy" in kwargs:
            self.strategy = kwargs.pop("strategy")
        if "win" in kwargs:
            self.win = kwargs.pop("win")
        _LegacyAgent.update(self, **kwargs)


cdef class OriginalEP(_LegacyOptimizer):
    """
    The original version of: Evolutionary Programming (EP)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import EP    >>> import numpy as np
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
    >>> model = EP.OriginalEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yao, X., Liu, Y. and Lin, G., 1999. Evolutionary programming made faster.
    IEEE Transactions on Evolutionary computation, 3(2), pp.82-102.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        bout_size: float = 0.05,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.bout_size = self.validator.check_float("bout_size", bout_size, (0, 1.0))
        self.set_parameters(["epoch", "pop_size", "bout_size"])
        self.sort_flag = True

    def initialize_variables(self):
        self.n_bout_size = int(self.bout_size * self.pop_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        strategy = self.generator.uniform(0, self.distance, self.problem.n_dims)
        times_win = 0
        return _OriginalEPAgent(solution=solution, strategy=strategy, win=times_win)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        child = []
        for idx in range(0, self.pop_size):
            pos_new = self.pop[idx].solution + self.pop[
                idx
            ].strategy * self.generator.normal(0, 1.0, self.problem.n_dims)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            s_old = (
                self.pop[idx].strategy
                + self.generator.normal(0, 1.0, self.problem.n_dims)
                * np.abs(self.pop[idx].strategy) ** 0.5
            )
            agent.update(solution=pos_new, strategy=s_old, win=0)
            child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                child[-1].target = self.get_target(pos_new)
        child = self.update_target_for_population(child)
        # Update the global best
        children = self.get_sorted_population(child, self.problem.minmax)
        pop = children + self.pop
        for i in range(0, len(pop)):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = self.generator.integers(0, len(pop))
                if self.compare_target(
                    pop[i].target, pop[rand_idx].target, self.problem.minmax
                ):
                    pop[i].win += 1
                else:
                    pop[rand_idx].win += 1
        pop = sorted(pop, key=lambda agent: agent.win, reverse=True)
        self.pop = pop[: self.pop_size]
