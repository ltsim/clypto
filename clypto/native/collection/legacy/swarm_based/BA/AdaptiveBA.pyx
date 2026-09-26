#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent
from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class _AdaptiveBAAgent(_LegacyAgent):
    cdef public object velocity
    cdef public object loudness
    cdef public object pulse_rate
    def __init__(self, solution=None, target=None, velocity=None, loudness=None, pulse_rate=None):
        _LegacyAgent.__init__(self, solution, target)
        self.velocity = velocity
        self.loudness = loudness
        self.pulse_rate = pulse_rate
    cpdef object copy(self):
        return _AdaptiveBAAgent(
            self.solution, None if self.target is None else self.target.copy(),
            self.velocity,
            self.loudness,
            self.pulse_rate,
        )
    def update(self, **kwargs):
        if "velocity" in kwargs:
            self.velocity = kwargs.pop("velocity")
        if "loudness" in kwargs:
            self.loudness = kwargs.pop("loudness")
        if "pulse_rate" in kwargs:
            self.pulse_rate = kwargs.pop("pulse_rate")
        _LegacyAgent.update(self, **kwargs)


cdef class AdaptiveBA(_LegacyOptimizer):
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
    >>> from clypto.native.collection.legacy.swarm_based import BA    >>> import numpy as np
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
        **kwargs: object
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
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.loudness_min = self.validator.check_float(
            "loudness_min", loudness_min, [0.5, 1.5]
        )
        self.loudness_max = self.validator.check_float(
            "loudness_max", loudness_max, [1.5, 3.0]
        )
        self.pr_min = self.validator.check_float("pr_min", pr_min, [-10.0, 10.0])
        self.pr_max = self.validator.check_float("pr_max", pr_max, [-10.0, 10.0])
        self.pf_min = self.validator.check_float("pf_min", pf_min, [-10.0, 10.0])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [0.0, 10.0])
        self.alpha = self.gamma = 0.9
        self.set_parameters(
            [
                "epoch",
                "pop_size",
                "loudness_min",
                "loudness_max",
                "pr_min",
                "pr_max",
                "pf_min",
                "pf_max",
            ]
        )
        self.sort_flag = False

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.lb, self.problem.ub)
        loudness = self.generator.uniform(self.loudness_min, self.loudness_max)
        pulse_rate = self.generator.uniform(self.pr_min, self.pr_max)
        return _AdaptiveBAAgent(
            solution=solution,
            velocity=velocity,
            loudness=loudness,
            pulse_rate=pulse_rate,
        )

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        mean_a = np.mean([agent.loudness for agent in self.pop])
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            pulse_frequency = self.generator.uniform(self.pf_min, self.pf_max)
            agent.velocity = agent.velocity + pulse_frequency * (
                self.pop[idx].solution - self.g_best.solution
            )
            x_new = self.pop[idx].solution + agent.velocity
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
                    pop_new[idx].target, self.pop[idx].target, self.problem.minmax
                )
                and self.generator.random() < pop_new[idx].loudness
            ):
                loudness = self.alpha * pop_new[idx].loudness
                pulse_rate = pop_new[idx].pulse_rate * (1 - np.exp(-self.gamma * epoch))
                self.pop[idx].update(
                    solution=pop_new[idx].solution,
                    target=pop_new[idx].target,
                    loudness=loudness,
                    pulse_rate=pulse_rate,
                )
