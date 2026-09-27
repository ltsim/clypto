#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.swarm_based.PSO.P_PSO cimport P_PSO
from clypto.native.collection.legacy.swarm_based.PSO._base cimport PSOPopulation


cdef class HPSO_TVAC(P_PSO):
    """
    The original version of: Hierarchical PSO Time-Varying Acceleration (HPSO-TVAC)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + ci (float): [0.3, 1.0], c initial, default = 0.5
        + cf (float): [0.0, 0.3], c final, default = 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.HPSO_TVAC(epoch=1000, pop_size=50, ci=0.5, cf=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Ghasemi, M., Aghaei, J. and Hadipour, M., 2017. New self-organising hierarchical PSO with
    jumping time-varying acceleration coefficients. Electronics Letters, 53(20), pp.1360-1362.
    """

    cdef public double cf
    cdef public double ci

    def __init__(self, epoch=10000, pop_size=100, ci=0.5, cf=0.1, **kwargs):
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            ci: c initial, default = 0.5
            cf: c final, default = 0.0
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=PSOPopulation)
        self.ci = cy.validator(float, ci, [0.3, 1.0], "ci")
        self.cf = cy.validator(float, cf, [0, 0.3], "cf")
        self.parameters = ["epoch", "pop_size", "ci", "cf"]
        self.sort_flag = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        c_it = ((self.cf - self.ci) * (epoch / self.epoch)) + self.ci
        for idx in range(0, pop_size):
            idx_k = self.generator.integers(0, pop_size)
            w = self.generator.normal()
            while np.abs(w - 1.0) < 0.01:
                w = self.generator.normal()
            c1_it = np.abs(w) ** (c_it * w)
            c2_it = np.abs(1 - w) ** (c_it / (1 - w))
            #################### HPSO
            v_new = c1_it * self.generator.uniform(0, 1, self.problem.n_dims) * (
                self.population[idx].pbest_solution - self.population[idx].solution
            ) + c2_it * self.generator.uniform(0, 1, self.problem.n_dims) * (
                self.g_best.solution
                + self.population[idx_k].pbest_solution
                - 2 * self.population[idx].solution
            )
            v_new = np.where(
                v_new == 0,
                np.sign(0.5 - self.generator.uniform())
                * self.generator.uniform()
                * self.v_max,
                v_new,
            )
            v_new = np.sign(v_new) * np.minimum(np.abs(v_new), self.v_max)
            #########################
            v_new = np.minimum(np.maximum(v_new, -self.v_max), self.v_max)
            pos_new = self.population[idx].solution + v_new
            pos_new = self.population.correct_solution(pos_new)
            self.population[idx].velocity = v_new
            candidate = self.population.evaluate_solution(pos_new)
            if cy.is_better(candidate, self.population[idx], self.problem.sense):
                self.population[idx].update_solution(candidate)
            self.population[idx].update_pbest(candidate, self.problem.sense)
