#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class GaussianSA(_LegacyOptimizer):
    """
    The developed version of: Gaussian Simulated Annealing (GaussianSA)

    Notes:
        + SA is single-based solution, so the pop_size parameter is not matter in this algorithm
        + The temp_init is very important factor. Should set it equal to the distance between LB and UB

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + cooling_rate (float): (0., 1.0), cooling rate, default=0.99
        + scale (float): (0., 100.), the scale in gaussian random, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import SA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = SA.GaussianSA(epoch=1000, pop_size=2, temp_init = 100, cooling_rate = 0.99, scale = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 2,
            temp_init: float = 100,
            cooling_rate: float = 0.99,
            scale: float = 0.1,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            cooling_rate (float): cooling rate, default=0.99
            scale (float): the scale in gaussian random, default=0.1
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [2, 10000])
        self.temp_init = self.validator.check_float("temp_init", temp_init, [1, 10000])
        self.cooling_rate = self.validator.check_float(
            "cooling_rate", cooling_rate, (0.0, 1.0)
        )
        self.scale = self.validator.check_float("scale", scale, (0.0, 100.0))
        self.set_parameters(["epoch", "temp_init", "cooling_rate", "scale"])

    def before_main_loop(self):
        # Initialize the system
        self.temp_current = self.temp_init
        self.agent_current = self.g_best.copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Perturb the current solution
        pos_new = self.agent_current.solution + self.generator.normal(
            scale=self.scale, size=self.problem.n_dims
        )
        agent = self.generate_agent(pos_new)
        # Accept or reject the new solution
        if self.compare_target(
                agent.target, self.agent_current.target, self.problem.minmax
        ):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(
                self.agent_current.target.fitness - agent.target.fitness
            )
            p_accept = np.exp(-delta_energy / self.temp_current)
            if self.generator.random() < p_accept:
                self.agent_current = agent
        # Reduce the temperature
        self.temp_current *= self.cooling_rate
        self.pop = [self.g_best.copy(), self.agent_current.copy()]
