#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.swarm_based.PSO._base cimport PSOPopulation


cdef class P_PSO(cy.Optimizer):
    """
    The original version of: Phasor Particle Swarm Optimization (P-PSO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.P_PSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Akbari, E., Rahimnejad, A., Razavi, S.E., Ghavidel, S. and Li, L., 2019.
    Phasor particle swarm optimization: a simple and efficient variant of PSO. Soft Computing, 23(19), pp.9701-9718.
    """

    def __init__(
        self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
        """
        super().__init__(**kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=PSOPopulation)
        self.parameters = ["epoch", "pop_size"]
        self.sort_flag = False

    def initialize_variables(self):
        pop_size = self.population.size()
        self.v_max = 0.5 * (self.problem.bounds.up - self.problem.bounds.low)
        self.dyn_delta_list = self.generator.uniform(0, 2 * np.pi, pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx, agent in enumerate(self.population.toarray()):
            aa = 2 * (np.sin(self.dyn_delta_list[idx]))
            bb = 2 * (np.cos(self.dyn_delta_list[idx]))
            ee = np.abs(np.cos(self.dyn_delta_list[idx])) ** aa
            tt = np.abs(np.sin(self.dyn_delta_list[idx])) ** bb
            v_new = ee * (
                agent.pbest_solution - agent.solution
            ) + tt * (self.g_best.solution - agent.solution)
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            agent.velocity = v_new
            x = agent.solution + v_new
            x = cy.correct_solution(self.problem, x)
            self.dyn_delta_list[idx] += np.abs(aa + bb) * (2 * np.pi)
            self.v_max = (np.abs(np.cos(self.dyn_delta_list[idx])) ** 2) * (
                self.problem.bounds.up - self.problem.bounds.low
            )
            candidate = self.population.evaluate_solution(x)
            if cy.is_better(candidate, agent, self.problem.sense):
                agent.update_solution(candidate)
            agent.update_pbest(candidate, self.problem.sense)
