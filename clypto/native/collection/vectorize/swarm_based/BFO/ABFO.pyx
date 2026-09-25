#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer._native.agent cimport _LegacyAgent


from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class ABFO(AgentListOptimizer):
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
    >>> from clypto.collection.swarm_based import BFO    >>> import numpy as np
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

    cdef public object N_adapt
    cdef public object N_split
    cdef public object C_s
    cdef public object Ped
    cdef public object C_e
    cdef public object p_eliminate
    cdef public object swim_length
    cdef public object Ns
    cdef public object support_parallel_modes

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
        *,
        name: str | None = None,
        mode: str | None = None,
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
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "C_s", "C_e", "Ped", "Ns", "N_adapt", "N_split"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.C_s = self.Ped = cy.validator(float, C_s, (0, 2.0), "C_s")
        self.C_e = self.Ped = cy.validator(float, C_e, (0, 1.0), "C_e")
        self.p_eliminate = self.Ped = cy.validator(float, Ped, (0, 1.0), "Ped")
        self.swim_length = self.Ns = cy.validator(int, Ns, [2, 100], "Ns")
        self.N_adapt = cy.validator(int, N_adapt, [0, 4], "N_adapt")
        self.N_split = cy.validator(int, N_split, [5, 50], "N_split")
        self.support_parallel_modes = False

    cdef void initialize_variables(self):
        self.C_s = self.C_s * (self.problem.ub - self.problem.lb)
        self.C_e = self.C_e * (self.problem.ub - self.problem.lb)

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        nutrients = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_solution = solution.copy()
        return FieldAgent(
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
            step_size / self.objs[idx].nutrients
            if self.objs[idx].nutrients > 0
            else step_size
        )
        return step_size

    def evolve_agents(self, epoch):
        for idx in range(0, self.pop_size):
            step_size = self.update_step_size__(self.objs, idx)
            for m in range(0, self.swim_length):  # Ns
                delta_i = (self.g_best.solution - self.objs[idx].solution) + (
                    self.objs[idx].local_solution - self.objs[idx].solution
                )
                delta = np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                unit_vector = (
                    self.generator.uniform(self.problem.lb, self.problem.ub)
                    if delta == 0
                    else (delta_i / delta)
                )
                pos_new = self.objs[idx].solution + step_size * unit_vector
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                if self.compare_target(
                    agent.target, self.objs[idx].target, self.problem.minmax
                ):
                    agent.nutrients += 1
                    self.objs[idx] = agent
                    # Update personal best
                    if self.compare_target(
                        agent.target, self.objs[idx].local_target, self.problem.minmax
                    ):
                        self.objs[idx].update(
                            local_solution=pos_new.copy(),
                            local_target=agent.target.copy(),
                        )
                else:
                    self.objs[idx].nutrients -= 1
            if self.objs[idx].nutrients > max(
                self.N_split,
                self.N_split + (len(self.objs) - self.pop_size) / self.N_adapt,
            ):
                tt = self.generator.normal(0, 1, self.problem.n_dims)
                pos_new = tt * self.objs[idx].solution + (1 - tt) * (
                    self.g_best.solution - self.objs[idx].solution
                )
                pos_new = self.correct_solution(pos_new)
                agent = self.generate_agent(pos_new)
                self.objs.append(agent)
            nut_min = min(
                self.N_adapt,
                self.N_adapt + (len(self.objs) - self.pop_size) / self.N_adapt,
            )
            if (
                self.objs[idx].nutrients < nut_min
                or self.generator.random() < self.p_eliminate
            ):
                self.objs[idx] = self.generate_agent()
        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(self.objs):
            if tuple(obj.solution.tolist()) in new_set:
                self.objs.pop(idx)
            else:
                new_set.add(tuple(obj.solution.tolist()))
        ## Balance the population by adding more agents or remove some agents
        n_agents = len(self.objs) - self.pop_size
        if n_agents < 0:
            for idx in range(0, n_agents):
                agent = self.generate_agent()
                self.objs.append(agent)
        elif n_agents > 0:
            list_idx_removed = self.generator.choice(
                range(0, len(self.objs)), n_agents, replace=False
            )
            pop_new = []
            for idx in range(0, len(self.objs)):
                if idx not in list_idx_removed:
                    pop_new.append(self.objs[idx])
            self.objs = pop_new
