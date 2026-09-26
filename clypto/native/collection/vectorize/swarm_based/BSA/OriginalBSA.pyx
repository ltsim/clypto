#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalBSA(AgentListOptimizer):
    """
    The original version of: Bird Swarm Algorithm (BSA)

    Links:
        1. https://doi.org/10.1080/0952813X.2015.1042530
        2. https://www.mathworks.com/matlabcentral/fileexchange/51256-bird-swarm-algorithm-bsa

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ff (int): (5, 20), flight frequency - default = 10
        + pff (float): the probability of foraging for food - default = 0.8
        + c_couples (list, tuple): [c1, c2] -> (2.0, 2.0), Cognitive accelerated coefficient, Social accelerated coefficient same as PSO
        + a_couples (list, tuple): [a1, a2] -> (1.5, 1.5), The indirect and direct effect on the birds' vigilance behaviours.
        + fc (float): (0.1, 1.0), The followed coefficient - default = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BSA    >>> import numpy as np
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
    >>> model = BSA.OriginalBSA(epoch=1000, pop_size=50, ff = 10, pff = 0.8, c1 = 1.5, c2 = 1.5, a1 = 1.0, a2 = 1.0, fc = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Meng, X.B., Gao, X.Z., Lu, L., Liu, Y. and Zhang, H., 2016. A new bio-inspired optimisation
    algorithm: Bird Swarm Algorithm. Journal of Experimental & Theoretical Artificial Intelligence, 28(4), pp.673-687.
    """

    cdef public object ff
    cdef public object pff
    cdef public object c1
    cdef public object c2
    cdef public object a1
    cdef public object a2
    cdef public object fc

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        ff: int = 10,
        pff: float = 0.8,
        c1: float = 1.5,
        c2: float = 1.5,
        a1: float = 1.0,
        a2: float = 1.0,
        fc: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            ff (int): flight frequency - default = 10
            pff (float): the probability of foraging for food - default = 0.8
            c1 (float): Cognitive accelerated coefficient same as PSO
            c2 (float): Social accelerated coefficient same as PSO
            a1 (float): The indirect effect on the birds' vigilance behaviours.
            a2 (float): The direct effect on the birds' vigilance behaviours.
            fc (float): The followed coefficient - default = 0.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "ff", "pff", "c1", "c2", "a1", "a2", "fc"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.ff = cy.validator(int, ff, [2, int(self.pop_size / 2)], "ff")
        self.pff = cy.validator(float, pff, (0, 1.0), "pff")
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.a1 = cy.validator(float, a1, (0, 5.0), "a1")
        self.a2 = cy.validator(float, a2, (0, 5.0), "a2")
        self.fc = cy.validator(float, fc, (0, 1.0), "fc")

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        local_position = solution.copy()
        return FieldAgent(solution=solution, local_solution=local_position)

    def generate_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        agent = self.generate_empty_agent(solution)
        agent.target = self.get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def evolve_agents(self, epoch):
        pos_list = np.array([agent.solution for agent in self.objs])
        fit_list = np.array([agent.local_target.fitness for agent in self.objs])
        pos_mean = np.mean(pos_list, axis=0)
        fit_sum = np.sum(fit_list)

        if epoch % self.ff != 0:
            pop_new = []
            for idx in range(0, self.pop_size):
                agent = self.objs[idx].copy()
                prob = (
                    self.generator.uniform() * 0.2 + self.pff
                )  # The probability of foraging for food
                if self.generator.uniform() < prob:  # Birds forage for food. Eq. 1
                    x_new = (
                        self.objs[idx].solution
                        + self.c1
                        * self.generator.uniform()
                        * (self.objs[idx].local_solution - self.objs[idx].solution)
                        + self.c2
                        * self.generator.uniform()
                        * (self.g_best.solution - self.objs[idx].solution)
                    )
                else:  # Birds keep vigilance. Eq. 2
                    A1 = self.a1 * np.exp(
                        -self.pop_size
                        * self.objs[idx].local_target.fitness
                        / (self.EPSILON + fit_sum)
                    )
                    k = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx})
                    )
                    t1 = (fit_list[idx] - fit_list[k]) / (
                        abs(fit_list[idx] - fit_list[k]) + self.EPSILON
                    )
                    A2 = self.a2 * np.exp(
                        t1 * self.pop_size * fit_list[k] / (fit_sum + self.EPSILON)
                    )
                    x_new = (
                        self.objs[idx].solution
                        + A1
                        * self.generator.uniform(0, 1)
                        * (pos_mean - self.objs[idx].solution)
                        + A2
                        * self.generator.uniform(-1, 1)
                        * (self.g_best.solution - self.objs[idx].solution)
                    )
                agent.solution = self.correct_solution(x_new)
                pop_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.target = self.get_target(agent.solution)
                    self.objs[idx] = self.get_better_agent(
                        agent, self.objs[idx], self.problem.minmax
                    )
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
                self.objs = self.greedy_selection_population(
                    self.objs, pop_new, self.problem.minmax
                )
        else:
            pop_new = self.objs.copy()
            # Divide the bird swarm into two parts: producers and scroungers.
            min_idx = np.argmin(fit_list)
            max_idx = np.argmax(fit_list)
            choose = 0
            if min_idx < 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 1
            if min_idx > 0.5 * self.pop_size and max_idx < 0.5 * self.pop_size:
                choose = 2
            if min_idx < 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 3
            if min_idx > 0.5 * self.pop_size and max_idx > 0.5 * self.pop_size:
                choose = 4

            if choose < 3:  # Producing (Equation 5)
                for idx in range(int(self.pop_size / 2 + 1), self.pop_size):
                    agent = self.objs[idx].copy()
                    x_new = (
                        self.objs[idx].solution
                        + self.generator.uniform(self.problem.lb, self.problem.ub)
                        * self.objs[idx].solution
                    )
                    agent.solution = self.correct_solution(x_new)
                    pop_new[idx] = agent
                if choose == 1:
                    x_new = (
                        self.objs[min_idx].solution
                        + self.generator.uniform(self.problem.lb, self.problem.ub)
                        * self.objs[min_idx].solution
                    )
                    agent = self.objs[min_idx].copy()
                    agent.solution = self.correct_solution(x_new)
                    pop_new[min_idx] = agent
                for i in range(0, int(self.pop_size / 2)):
                    if choose == 2 or min_idx != i:
                        agent = self.objs[i].copy()
                        FL = self.generator.uniform() * 0.4 + self.fc
                        idx = self.generator.integers(
                            0.5 * self.pop_size + 1, self.pop_size
                        )
                        x_new = (
                            self.objs[i].solution
                            + (self.objs[idx].solution - self.objs[i].solution) * FL
                        )
                        agent.solution = self.correct_solution(x_new)
                        pop_new[i] = agent
            else:  # Scrounging (Equation 6)
                for i in range(0, int(0.5 * self.pop_size)):
                    agent = self.objs[i].copy()
                    x_new = (
                        self.objs[i].solution
                        + self.generator.uniform(self.problem.lb, self.problem.ub)
                        * self.objs[i].solution
                    )
                    agent.solution = self.correct_solution(x_new)
                    pop_new[i] = agent
                if choose == 4:
                    agent = self.objs[min_idx].copy()
                    x_new = (
                        self.objs[min_idx].solution
                        + self.generator.uniform(self.problem.lb, self.problem.ub)
                        * self.objs[min_idx].solution
                    )
                    agent.solution = self.correct_solution(x_new)
                for i in range(int(self.pop_size / 2 + 1), self.pop_size):
                    if choose == 3 or min_idx != i:
                        agent = self.objs[i].copy()
                        FL = self.generator.uniform() * 0.4 + self.fc
                        idx = self.generator.integers(0, 0.5 * self.pop_size)
                        x_new = (
                            self.objs[i].solution
                            + (self.objs[idx].solution - self.objs[i].solution) * FL
                        )
                        agent.solution = self.correct_solution(x_new)
                        pop_new[i] = agent
            if self.mode in self.AVAILABLE_MODES:
                pop_new = self.update_target_for_population(pop_new)
            else:
                for idx in range(0, self.pop_size):
                    pop_new[idx].target = self.get_target(pop_new[idx].solution)
            self.objs = self.greedy_selection_population(
                self.objs, pop_new, self.problem.minmax
            )
