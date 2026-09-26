#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class _OriginalBAAgent(LegacyAgent):
    cdef public object velocity
    cdef public object pulse_frequency


cdef class OriginalBA(LegacyOptimizer):
    """
    The original version of: Bat-inspired Algorithm (BA)

    Notes
    ~~~~~
    + The value of A and r parameters are constant

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness (float): (1.0, 2.0), loudness, default = 0.8
        + pulse_rate (float): (0.15, 0.85), pulse rate / emission rate, default = 0.95
        + pulse_frequency (list, tuple): (pf_min, pf_max) -> ([0, 3], [5, 20]), pulse frequency, default = (0, 10)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BA    >>> import numpy as np
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
    >>> model = BA.OriginalBA(epoch=1000, pop_size=50, loudness=0.8, pulse_rate=0.95, pf_min=0.1, pf_max=10.0)
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
        pop_size: int = 100,
        loudness: float = 0.8,
        pulse_rate: float = 0.95,
        pf_min: float = 0.0,
        pf_max: float = 10.0,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness (float): (A_min, A_max): loudness, default = 0.8
            pulse_rate (float): (r_min, r_max): pulse rate / emission rate, default = 0.95
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.loudness = self.validator.check_float("loudness", loudness, (0, 1.0))
        self.pulse_rate = self.validator.check_float("pulse_rate", pulse_rate, (0, 1.0))
        self.pf_min = self.validator.check_float("pf_min", pf_min, [0.0, 3.0])
        self.pf_max = self.validator.check_float("pf_max", pf_max, [5.0, 20.0])
        self._set_parameters(
            ["epoch", "pop_size", "loudness", "pulse_rate", "pf_min", "pf_max"]
        )
        self.alpha = self.gamma = 0.9
        self.sort_flag = False

    def _generate_empty_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        pulse_frequency = (
            self.pf_min + (self.pf_max - self.pf_min) * self.generator.uniform()
        )
        return _OriginalBAAgent(
            solution=solution, velocity=velocity, pulse_frequency=pulse_frequency
        )

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            vec = agent.velocity + self.pop[idx].pulse_frequency * (
                self.pop[idx].solution - self.g_best.solution
            )
            x_new = self.pop[idx].solution + agent.velocity
            ## Local Search around g_best position
            if self.generator.random() > self.pulse_rate:
                x_new = self.g_best.solution + 0.001 * self.generator.normal(
                    self.problem.n_dims
                )
            pos_new = self._correct_solution(x_new)
            agent.update(solution=pos_new, velocity=vec)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self._get_target(pos_new)
        pop_new = self._update_target_for_population(pop_new)
        for idx in range(self.pop_size):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if (
                self._compare_target(
                    pop_new[idx].target, self.pop[idx].target, self.problem.sense
                )
                and self.generator.random() < self.loudness
            ):
                self.pop[idx].update(
                    solution=pop_new[idx].solution, target=pop_new[idx].target
                )
