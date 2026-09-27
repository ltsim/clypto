#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class GaussianSA(cy.Optimizer):
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
    >>> model = SA.GaussianSA(epoch=1000, pop_size=2, temp_init = 100, cooling_rate = 0.99, scale = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double cooling_rate
    cdef public double scale
    cdef public double temp_init

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
        super().__init__(parameters=["epoch", "temp_init", "cooling_rate", "scale"], **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[2, 10000])
        self.temp_init = cy.validator(float, temp_init, [1, 10000], "temp_init")
        self.cooling_rate = cy.validator(float, cooling_rate, (0.0, 1.0), "cooling_rate")
        self.scale = cy.validator(float, scale, (0.0, 100.0), "scale")

    def before_main_loop(self):
        # Initialize the system
        self.temp_current = self.temp_init
        self.agent_current = self.g_best.copy()

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Perturb the current solution
        pos_new = self.agent_current.solution + self.generator.normal(
            scale=self.scale, size=self.problem.n_dims
        )
        agent = self.population.generate_agent(pos_new)
        # Accept or reject the new solution
        if cy.is_better(agent, self.agent_current, self.problem.sense):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(
                self.agent_current.fitness - agent.fitness
            )
            p_accept = np.exp(-delta_energy / self.temp_current)
            if self.generator.random() < p_accept:
                self.agent_current = agent
        # Reduce the temperature
        self.temp_current *= self.cooling_rate
        self.population = [self.g_best.copy(), self.agent_current.copy()]
