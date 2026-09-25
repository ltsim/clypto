#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
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


cdef class AdaptiveBA(AgentListOptimizer):
    """
    The original version of: Adaptive Bat-inspired Algorithm (ABA)

    Notes
    ~~~~~
    + The value of A and r are changing after each iteration

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness_min (float): A_min - loudness, default=1.0
        + loudness_max (float): A_max - loudness, default=2.0
        + pr_min (float): pulse rate / emission rate min, default = 0.15
        + pr_max (float): pulse rate / emission rate max, default = 0.85
        + pf_min (float): pulse frequency min, default = 0
        + pf_max (float): pulse frequency max, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BA    >>> import numpy as np
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
    >>> model = BA.AdaptiveBA(epoch=1000, pop_size=50, loudness_min = 1.0, loudness_max = 2.0, pr_min = -2.5, pr_max = 0.85, pf_min = 0.1, pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    cdef public object loudness_min
    cdef public object loudness_max
    cdef public object pr_min
    cdef public object pr_max
    cdef public object pf_min
    cdef public object pf_max
    cdef public object alpha
    cdef public object gamma

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: object = 100,
        loudness_min: float = 1.0,
        loudness_max: float = 2.0,
        pr_min: float = 0.15,
        pr_max: float = 0.85,
        pf_min: float = -10.0,
        pf_max: float = 10.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness_min (float): A_min - loudness, default=1.0
            loudness_max (float): A_max - loudness, default=2.0
            pr_min (float): pulse rate / emission rate min, default = 0.15
            pr_max (float): pulse rate / emission rate max, default = 0.85
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "loudness_min",
                "loudness_max",
                "pr_min",
                "pr_max",
                "pf_min",
                "pf_max",
            ],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.loudness_min = cy.validator(float, loudness_min, [0.5, 1.5], "loudness_min")
        self.loudness_max = cy.validator(float, loudness_max, [1.5, 3.0], "loudness_max")
        self.pr_min = cy.validator(float, pr_min, [-10.0, 10.0], "pr_min")
        self.pr_max = cy.validator(float, pr_max, [-10.0, 10.0], "pr_max")
        self.pf_min = cy.validator(float, pf_min, [-10.0, 10.0], "pf_min")
        self.pf_max = cy.validator(float, pf_max, [0.0, 10.0], "pf_max")
        self.alpha = self.gamma = 0.9

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        loudness = self.generator.uniform(self.loudness_min, self.loudness_max)
        pulse_rate = self.generator.uniform(self.pr_min, self.pr_max)
        return FieldAgent(
            solution=solution,
            velocity=velocity,
            loudness=loudness,
            pulse_rate=pulse_rate,
        )

    def evolve_agents(self, epoch):
        mean_a = np.mean([agent.loudness for agent in self.objs])
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.objs[idx].copy()
            pulse_frequency = self.generator.uniform(self.pf_min, self.pf_max)
            agent.velocity = agent.velocity + pulse_frequency * (
                self.objs[idx].solution - self.g_best.solution
            )
            x_new = self.objs[idx].solution + agent.velocity
            ## Local Search around g_best position
            if self.generator.random() > agent.pulse_rate:
                x_new = self.g_best.solution + mean_a * self.generator.normal(-1, 1)
            pos_new = self.correct_solution(x_new)
            agent.solution = pos_new
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        for idx in range(0, self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if (
                self.compare_target(
                    pop_new[idx].target, self.objs[idx].target, self.problem.minmax
                )
                and self.generator.random() < pop_new[idx].loudness
            ):
                loudness = self.alpha * pop_new[idx].loudness
                pulse_rate = pop_new[idx].pulse_rate * (1 - np.exp(-self.gamma * epoch))
                self.objs[idx].update(
                    solution=pop_new[idx].solution,
                    target=pop_new[idx].target,
                    loudness=loudness,
                    pulse_rate=pulse_rate,
                )
