#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer._native.agent cimport _LegacyAgent
from clypto.optimizer._native.legacy cimport _LegacyOptimizer


cdef class _P_PSOAgent(_LegacyAgent):
    cdef public object velocity
    cdef public object local_solution
    cdef public object local_target
    def __init__(self, solution=None, target=None, velocity=None, local_solution=None, local_target=None):
        _LegacyAgent.__init__(self, solution, target)
        self.velocity = velocity
        self.local_solution = local_solution
        self.local_target = local_target
    cpdef object copy(self):
        return _P_PSOAgent(
            self.solution, None if self.target is None else self.target.copy(),
            self.velocity,
            self.local_solution,
            self.local_target,
        )
    def update(self, **kwargs):
        if "velocity" in kwargs:
            self.velocity = kwargs.pop("velocity")
        if "local_solution" in kwargs:
            self.local_solution = kwargs.pop("local_solution")
        if "local_target" in kwargs:
            self.local_target = kwargs.pop("local_target")
        _LegacyAgent.update(self, **kwargs)


cdef class P_PSO(_LegacyOptimizer):
    """
    The original version of: Phasor Particle Swarm Optimization (P-PSO)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import PSO    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = PSO.P_PSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Akbari, E., Rahimnejad, A., Razavi, S.E., Ghavidel, S. and Li, L., 2019.
    Phasor particle swarm optimization: a simple and efficient variant of PSO. Soft Computing, 23(19), pp.9701-9718.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False
        self.is_parallelizable = False

    def initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.dyn_delta_list = self.generator.uniform(0, 2 * np.pi, self.pop_size)

    def generate_empty_agent(self, solution: np.ndarray | None = None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-self.v_max, self.v_max)
        local_pos = solution.copy()
        return _P_PSOAgent(
            solution=solution, velocity=velocity, local_solution=local_pos
        )

    def generate_agent(self, solution: np.ndarray | None = None):
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            aa = 2 * (np.sin(self.dyn_delta_list[idx]))
            bb = 2 * (np.cos(self.dyn_delta_list[idx]))
            ee = np.abs(np.cos(self.dyn_delta_list[idx])) ** aa
            tt = np.abs(np.sin(self.dyn_delta_list[idx])) ** bb
            v_new = ee * (
                self.pop[idx].local_solution - self.pop[idx].solution
            ) + tt * (self.g_best.solution - self.pop[idx].solution)
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            self.pop[idx].velocity = v_new
            pos_new = self.pop[idx].solution + v_new
            pos_new = self.correct_solution(pos_new)
            self.dyn_delta_list[idx] += np.abs(aa + bb) * (2 * np.pi)
            self.v_max = (np.abs(np.cos(self.dyn_delta_list[idx])) ** 2) * (
                self.problem.ub - self.problem.lb
            )
            target = self.get_target(pos_new)
            if self.compare_target(target, self.pop[idx].target, self.problem.minmax):
                self.pop[idx].update(solution=pos_new.copy(), target=target.copy())
            if self.compare_target(
                target, self.pop[idx].local_target, self.problem.minmax
            ):
                self.pop[idx].update(
                    local_solution=pos_new.copy(), local_target=target.copy()
                )
