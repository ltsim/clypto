#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget
from clypto.optimizer.native.agent cimport LegacyNativeAgent


cdef class GaussianSA(LegacyNativeOptimizer):
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
    >>> from clypto.native.collection.vectorize.physics_based import SA    >>> import numpy as np
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

    cdef public object temp_init
    cdef public object cooling_rate
    cdef public object scale
    cdef public object temp_current
    cdef public object agent_current

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 2,
        temp_init: float = 100,
        cooling_rate: float = 0.99,
        scale: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            cooling_rate (float): cooling rate, default=0.99
            scale (float): the scale in gaussian random, default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "temp_init", "cooling_rate", "scale"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [2, 10000], "pop_size")
        self.temp_init = cy.validator(float, temp_init, [1, 10000], "temp_init")
        self.cooling_rate = cy.validator(float, cooling_rate, (0.0, 1.0), "cooling_rate")
        self.scale = cy.validator(float, scale, (0.0, 100.0), "scale")

    cdef void before_main_loop(self):
        # Initialize the system
        self.temp_current = self.temp_init
        self.agent_current = self.g_best.copy()

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef object cur = self.agent_current
        # Perturb the current solution
        pos_new = cur.solution + self.generator.normal(scale=self.scale, size=self.problem.n_dims)
        tar = self.get_target(pos_new)
        agent = LegacyNativeAgent(pos_new, tar)
        # Accept or reject the new solution
        if self.compare_fitness(tar.fitness, cur.target.fitness, self.problem.minmax):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(cur.target.fitness - tar.fitness)
            p_accept = np.exp(-delta_energy / self.temp_current)
            if self.generator.random() < p_accept:
                self.agent_current = agent
        # Reduce the temperature
        self.temp_current *= self.cooling_rate
        self.pop = self.two_agents__(self.g_best, self.agent_current)

    def two_agents__(self, a, b):
        """A population made of two agents (the best so far and the current one)."""
        cdef NativePopulation pop = self.pop.take(np.zeros(2, dtype=int))
        ops.set_row(pop, 0, a.solution, a.target)
        ops.set_row(pop, 1, b.solution, b.target)
        return pop
