#!/usr/bin/env python
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class _ABFOAgent(LegacyAgent):
    cdef public object nutrients
    cdef public object local_solution
    cdef public object local_target


cdef class ABFO(LegacyOptimizer):
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
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.C_s = self.Ped = self.validator.check_float("C_s", C_s, (0, 2.0))
        self.C_e = self.Ped = self.validator.check_float("C_e", C_e, (0, 1.0))
        self.p_eliminate = self.Ped = self.validator.check_float("Ped", Ped, (0, 1.0))
        self.swim_length = self.Ns = self.validator.check_int("Ns", Ns, [2, 100])
        self.N_adapt = self.validator.check_int("N_adapt", N_adapt, [0, 4])
        self.N_split = self.validator.check_int("N_split", N_split, [5, 50])
        self._set_parameters(
            ["epoch", "pop_size", "C_s", "C_e", "Ped", "Ns", "N_adapt", "N_split"]
        )
        self.support_parallel_modes = False
        self.sort_flag = False

    def _initialize_variables(self):
        self.C_s = self.C_s * (self.problem.bounds.up - self.problem.bounds.low)
        self.C_e = self.C_e * (self.problem.bounds.up - self.problem.bounds.low)

    def _generate_empty_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        nutrients = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_solution = solution.copy()
        return _ABFOAgent(
            solution=solution, nutrients=nutrients, local_solution=local_solution
        )

    def _generate_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        agent = self._generate_empty_agent(solution)
        agent.target = self._get_target(agent.solution)
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

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

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
                    self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
                    if delta == 0
                    else (delta_i / delta)
                )
                pos_new = self.pop[idx].solution + step_size * unit_vector
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_agent(pos_new)
                if self._compare_target(
                    agent.target, self.pop[idx].target, self.problem.sense
                ):
                    agent.nutrients += 1
                    self.pop[idx] = agent
                    # Update personal best
                    if self._compare_target(
                        agent.target, self.pop[idx].local_target, self.problem.sense
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
                pos_new = self._correct_solution(pos_new)
                agent = self._generate_agent(pos_new)
                self.pop.append(agent)
            nut_min = min(
                self.N_adapt,
                self.N_adapt + (len(self.pop) - self.pop_size) / self.N_adapt,
            )
            if (
                self.pop[idx].nutrients < nut_min
                or self.generator.random() < self.p_eliminate
            ):
                self.pop[idx] = self._generate_agent()
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
                agent = self._generate_agent()
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
