#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class _AIW_PSOAgent(LegacyAgent):
    cdef public object velocity
    cdef public object local_solution
    cdef public object local_target


cdef class AIW_PSO(LegacyOptimizer):
    """
    The original version of: Adaptive Inertia Weight Particle Swarm Optimization (AIW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + alpha (float): [0., 1.0], The positive constant, default = 0.4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import PSO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = PSO.AIW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, alpha=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, Z., Yu, F., Shi, Z., Wang, Y. (2006). Adaptive Inertia Weight Particle Swarm Optimization. In: Rutkowski, L.,
    Tadeusiewicz, R., Zadeh, L.A., Żurada, J.M. (eds) Artificial Intelligence and Soft Computing – ICAISC 2006. ICAISC 2006.
    Lecture Notes in Computer Science(), vol 4029. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11785231_48
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        alpha: float = 0.4,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            alpha: The positive constant, default = 0.4
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c1 = self.validator.check_float("c1", c1, (0, 5.0))
        self.c2 = self.validator.check_float("c2", c2, (0, 5.0))
        self.alpha = self.validator.check_float("alpha", alpha, [0.0, 1.0])
        self._set_parameters(["epoch", "pop_size", "c1", "c2", "alpha"])
        self.sort_flag = False

    def _initialize_variables(self):
        self.v_max = 0.5 * (self.problem.bounds.up - self.problem.bounds.low)
        self.v_min = -self.v_max

    def _generate_empty_agent(self, solution: np.ndarray | None = None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.v_min, self.v_max)
        local_pos = solution.copy()
        return _AIW_PSOAgent(
            solution=solution, velocity=velocity, local_solution=local_pos
        )

    def _generate_agent(self, solution: np.ndarray | None = None):
        agent = self._generate_empty_agent(solution)
        agent.target = self._get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def _amend_solution(self, solution: np.ndarray) -> np.ndarray:
        condition = np.logical_and(
            self.problem.bounds.low <= solution, solution <= self.problem.bounds.up
        )
        pos_rand = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        return np.where(condition, solution, pos_rand)

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        current_best = self._get_best_agent(self.pop, self.problem.sense)
        for idx in range(0, self.pop_size):
            denom = np.abs(self.pop[idx].local_solution - current_best.solution)
            denom = np.where(denom == 0, 1e-6, denom)
            isa = (
                np.abs(self.pop[idx].solution - self.pop[idx].local_solution) / denom
            )  # individual search ability
            w = 1 - self.alpha * (1.0 / (1.0 + np.exp(-isa)))
            cognitive = (
                self.c1
                * self.generator.random(self.problem.n_dims)
                * (self.pop[idx].local_solution - self.pop[idx].solution)
            )
            social = (
                self.c2
                * self.generator.random(self.problem.n_dims)
                * (self.g_best.solution - self.pop[idx].solution)
            )
            velocity = w * self.pop[idx].velocity + cognitive + social
            self.pop[idx].velocity = np.clip(velocity, self.v_min, self.v_max)
            pos_new = self.pop[idx].solution + self.pop[idx].velocity
            pos_new = self._correct_solution(pos_new)
            target = self._get_target(pos_new)
            if self._compare_target(target, self.pop[idx].target, self.problem.sense):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self._compare_target(
                target, self.pop[idx].local_target, self.problem.sense
            ):
                self.pop[idx].update(
                    local_solution=pos_new.copy(), local_target=target.copy()
                )
