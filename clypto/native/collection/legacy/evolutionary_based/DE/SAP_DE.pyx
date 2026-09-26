#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class _SAP_DEAgent(LegacyAgent):
    cdef public object crossover
    cdef public object mutation
    cdef public object pop_size


cdef class SAP_DE(LegacyOptimizer):
    """
    The original version of: Differential Evolution with Self-Adaptive Populations (SAP_DE)

    Links:
        1. https://doi.org/10.1007/s00500-005-0537-1

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + branch (str): ["ABS" or "REL"], gaussian (absolute) or uniform (relative) method

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.SAP_DE(epoch=1000, pop_size=50, branch = "ABS")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Teo, J., 2006. Exploring dynamic self-adaptive populations in differential evolution. Soft Computing, 10(8), pp.673-686.
    """

    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 100,
        branch: str = "ABS",
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            branch (str): gaussian (absolute) or uniform (relative) method
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.branch = self.validator.check_str("branch", branch, ["ABS", "REL"])
        self._set_parameters(["epoch", "pop_size", "branch"])
        self.fixed_pop_size = self.pop_size
        self.sort_flag = False

    def _generate_empty_agent(self, solution: np.ndarray | None = None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        crossover_rate = self.generator.uniform(0, 1)
        mutation_rate = self.generator.uniform(0, 1)
        if self.branch == "ABS":
            pop_size = int(10 * self.problem.n_dims + self.generator.normal(0, 1))
        else:  # elif self.branch == "REL":
            pop_size = int(10 * self.problem.n_dims + self.generator.uniform(-0.5, 0.5))

        return _SAP_DEAgent(
            solution=solution,
            crossover=crossover_rate,
            mutation=mutation_rate,
            pop_size=pop_size,
        )

    def edit_to_range__(self, var=None, lower=0, upper=1, func_value=None):
        while var <= lower or var >= upper:
            if var <= lower:
                var += func_value()
            if var >= upper:
                var -= func_value()
        return var

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        for idx in range(0, self.pop_size):
            # Choose 3 random element and different to idx
            idxs = self.generator.choice(
                list(set(range(0, self.pop_size)) - {idx}), 3, replace=False
            )
            j = self.generator.integers(0, self.pop_size)
            self.F = self.generator.normal(0, 1)
            ## Crossover
            if self.generator.uniform(0, 1) < self.pop[idx].crossover or idx == j:
                pos_new = self.pop[idxs[0]].solution + self.F * (
                    self.pop[idxs[1]].solution - self.pop[idxs[2]].solution
                )
                cr_new = self.pop[idxs[0]].crossover + self.F * (
                    self.pop[idxs[1]].crossover - self.pop[idxs[2]].crossover
                )
                mr_new = self.pop[idxs[0]].mutation + self.F * (
                    self.pop[idxs[1]].mutation - self.pop[idxs[2]].mutation
                )
                if self.branch == "ABS":
                    ps_new = self.pop[idxs[0]].pop_size + int(
                        self.F
                        * (self.pop[idxs[1]].pop_size - self.pop[idxs[2]].pop_size)
                    )
                else:  # elif self.branch == "REL":
                    ps_new = self.pop[idxs[0]].pop_size + self.F * (
                        self.pop[idxs[1]].pop_size - self.pop[idxs[2]].pop_size
                    )
                pos_new = self._correct_solution(pos_new)
                cr_new = self.edit_to_range__(cr_new, 0, 1, self.generator.random)
                mr_new = self.edit_to_range__(mr_new, 0, 1, self.generator.random)
                agent = self._generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self._get_target(pos_new)
                    agent.update(crossover=cr_new, mutation=mr_new, pop_size=ps_new)
            else:
                pop.append(self.pop[idx].copy())
            ## Mutation
            if self.generator.uniform(0, 1) < self.pop[idxs[0]].mutation:
                pos_new = self.pop[idx].solution + self.generator.normal(
                    0, self.pop[idxs[0]].mutation
                )
                cr_new = self.generator.normal(0, 1)
                mr_new = self.generator.normal(0, 1)
                if self.branch == "ABS":
                    ps_new = self.pop[idx].pop_size + int(self.generator.normal(0.5, 1))
                else:  # elif self.branch == "REL":
                    ps_new = self.pop[idx].pop_size + self.generator.normal(
                        0, self.pop[idxs[0]].mutation
                    )
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_empty_agent(pos_new)
                pop.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self._get_target(pos_new)
                    agent.update(crossover=cr_new, mutation=mr_new, pop_size=ps_new)
        pop = self._update_target_for_population(pop)
        # Calculate new population size
        total = np.sum([pop[idx].pop_size for idx in range(0, self.pop_size)])
        if self.branch == "ABS":
            m_new = int(total / self.pop_size)
        else:  # elif self.branch == "REL":
            m_new = int(self.pop_size + total)
        if m_new <= 4:
            m_new = self.fixed_pop_size + int(self.generator.uniform(0, 4))
        elif m_new > 4 * self.fixed_pop_size:
            m_new = self.fixed_pop_size - int(self.generator.uniform(0, 4))
        ## Change population by population size
        if m_new <= self.pop_size:
            self.pop = pop[:m_new]
        else:
            pop_sorted = self._get_sorted_population(pop, self.problem.sense)
            self.pop = pop + pop_sorted[: m_new - self.pop_size]
        self.pop_size = len(self.pop)
