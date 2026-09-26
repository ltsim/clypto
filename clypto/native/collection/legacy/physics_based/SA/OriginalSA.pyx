#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalSA(LegacyOptimizer):
    """
    The original version of: Simulated Annealing (SA)

    Notes:
        + SA is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + step_size (float): the step size of random movement, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import SA    >>> import numpy as np
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
    >>> model = SA.OriginalSA(epoch=1000, pop_size=50, temp_init = 100, step_size = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 2,
            temp_init: float = 100,
            step_size: float = 0.1,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            step_size (float): the step size of random movement, default=0.1
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.temp_init = self.validator.check_float("temp_init", temp_init, [1, 10000])
        self.step_size = self.validator.check_float(
            "step_size", step_size, (-100.0, 100.0)
        )
        self._set_parameters(["epoch", "temp_init", "step_size"])

    def _before_main_loop(self):
        self.agent_current = self.g_best.copy()

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Perturb the current solution

        pos_new = (
                self.agent_current.solution
                + self.generator.standard_normal(self.problem.n_dims) * self.step_size
        )
        agent = self._generate_agent(pos_new)
        # Accept or reject the new solution
        if self._compare_target(
                agent.target, self.agent_current.target, self.problem.sense
        ):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(
                self.agent_current.target.fitness - agent.target.fitness
            )
            # calculate probability acceptance criterion
            p_accept = np.exp(-delta_energy / (self.temp_init / epoch))
            if self.generator.random() < p_accept:
                self.agent_current = agent
        self.pop = [self.g_best.copy(), self.agent_current.copy()]
