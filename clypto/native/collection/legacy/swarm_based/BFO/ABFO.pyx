#!/usr/bin/env python
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent
from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class _ABFOAgent(_LegacyAgent):
    cdef public object nutrients
    cdef public object local_solution
    cdef public object local_target
    def __init__(self, solution=None, target=None, nutrients=None, local_solution=None, local_target=None):
        _LegacyAgent.__init__(self, solution, target)
        self.nutrients = nutrients
        self.local_solution = local_solution
        self.local_target = local_target
    cpdef object copy(self):
        return _ABFOAgent(
            self.solution, None if self.target is None else self.target.copy(),
            self.nutrients,
            self.local_solution,
            self.local_target,
        )
    def update(self, **kwargs):
        if "nutrients" in kwargs:
            self.nutrients = kwargs.pop("nutrients")
        if "local_solution" in kwargs:
            self.local_solution = kwargs.pop("local_solution")
        if "local_target" in kwargs:
            self.local_target = kwargs.pop("local_target")
        _LegacyAgent.update(self, **kwargs)


cdef class ABFO(_LegacyOptimizer):
    """
    The original version of: Adaptive Bacterial Foraging Optimization (ABFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:

        + C_s (float): step size start, default=0.1
        + C_e (float): step size end, default=0.001
        + Ped (float): Probability eliminate, default=0.01
        + Ns (int): swim_length, default=4
        + N_adapt (int): Dead threshold value default=2
        + N_split (int): Split threshold value, default=40

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BFO    >>> import numpy as np
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
    >>> model = BFO.ABFO(epoch=1000, pop_size=50, C_s=0.1, C_e=0.001, Ped = 0.01, Ns = 4, N_adapt = 2, N_split = 40)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019, April. Building resource auto-scaler with functional-link
    neural network and adaptive bacterial foraging optimization. In International Conference on
    Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        C_s: float = 0.1,
        C_e: float = 0.001,
        Ped: float = 0.01,
        Ns: int = 4,
        N_adapt: int = 2,
        N_split: int = 40,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            C_s (float): step size start, default=0.1
            C_e (float): step size end, default=0.001
            Ped (float): Probability eliminate, default=0.01
            Ns (int): swim_length, default=4
            N_adapt (int): Dead threshold value default=2
            N_split (int): Split threshold value, default=40
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.C_s = self.Ped = self.validator.check_float("C_s", C_s, (0, 2.0))
        self.C_e = self.Ped = self.validator.check_float("C_e", C_e, (0, 1.0))
        self.p_eliminate = self.Ped = self.validator.check_float("Ped", Ped, (0, 1.0))
        self.swim_length = self.Ns = self.validator.check_int("Ns", Ns, [2, 100])
        self.N_adapt = self.validator.check_int("N_adapt", N_adapt, [0, 4])
        self.N_split = self.validator.check_int("N_split", N_split, [5, 50])
        self.set_parameters(
            ["epoch", "pop_size", "C_s", "C_e", "Ped", "Ns", "N_adapt", "N_split"]
        )
        self.support_parallel_modes = False
        self.sort_flag = False

    def initialize_variables(self):
        self.C_s = self.C_s * (self.problem.ub - self.problem.lb)
        self.C_e = self.C_e * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        nutrients = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_solution = solution.copy()
        return _ABFOAgent(
            solution=solution, nutrients=nutrients, local_solution=local_solution
        )

    def generate_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def update_step_size__(self, pop=None, idx=None):
        total_fitness = np.sum([agent.target.fitness for agent in pop])
        step_size = (
            self.C_s - (self.C_s - self.C_e) * pop[idx].target.fitness / total_fitness
        )
        step_size = (
            step_size / self.pop[idx].nutrients
            if self.pop[idx].nutrients > 0
            else step_size
        )
        return step_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            step_size = self.update_step_size__(self.pop, idx)
            for m in range(0, self.swim_length):  # Ns
                delta_i = (self.g_best.solution - self.pop[idx].solution) + (
                    self.pop[idx].local_solution - self.pop[idx].solution
                )
                delta = np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                unit_vector = (
                    self.generator.uniform(self.problem.lb, self.problem.ub)
                    if delta == 0
                    else (delta_i / delta)
                )
                pos_new = self.pop[idx].solution + step_size * unit_vector
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(
                    agent.target, self.pop[idx].target, self.problem.minmax
                ):
                    agent.nutrients += 1
                    self.pop[idx] = agent
                    # Update personal best
                    if self.compare_target(
                        agent.target, self.pop[idx].local_target, self.problem.minmax
                    ):
                        self.pop[idx].update(
                            local_solution=pos_new.copy(),
                            local_target=agent.target.copy(),
                        )
                else:
                    self.pop[idx].nutrients -= 1
            if self.pop[idx].nutrients > max(
                self.N_split,
                self.N_split + (len(self.pop) - self.pop_size) / self.N_adapt,
            ):
                tt = self.generator.normal(0, 1, self.problem.n_dims)
                pos_new = tt * self.pop[idx].solution + (1 - tt) * (
                    self.g_best.solution - self.pop[idx].solution
                )
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                self.pop.append(agent)
            nut_min = min(
                self.N_adapt,
                self.N_adapt + (len(self.pop) - self.pop_size) / self.N_adapt,
            )
            if (
                self.pop[idx].nutrients < nut_min
                or self.generator.random() < self.p_eliminate
            ):
                self.pop[idx] = self.generate_agent()
        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(self.pop):
            if tuple(obj.solution.tolist()) in new_set:
                self.pop.pop(idx)
            else:
                new_set.add(tuple(obj.solution.tolist()))
        ## Balance the population by adding more agents or remove some agents
        n_agents = len(self.pop) - self.pop_size
        if n_agents < 0:
            for idx in range(0, n_agents):
                agent = self.generate_agent()
                self.pop.append(agent)
        elif n_agents > 0:
            list_idx_removed = self.generator.choice(
                range(0, len(self.pop)), n_agents, replace=False
            )
            pop_new = []
            for idx in range(0, len(self.pop)):
                if idx not in list_idx_removed:
                    pop_new.append(self.pop[idx])
            self.pop = pop_new
