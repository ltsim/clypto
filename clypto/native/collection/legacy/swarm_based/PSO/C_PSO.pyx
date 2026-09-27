#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.legacy.swarm_based.PSO.P_PSO cimport P_PSO
from clypto.native.collection.legacy.swarm_based.PSO._base cimport PSOPopulation


cdef class C_PSO(P_PSO):
    """
    The original version of: Chaos Particle Swarm Optimization (C-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1.0, 3.0] local coefficient, default = 2.05
        + c2 (float): [1.0, 3.0] global coefficient, default = 2.05
        + w_min (float): [0.1, 0.4], Weight min of bird, default = 0.4
        + w_max (float): [0.4, 2.0], Weight max of bird, default = 0.9

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
    >>> model = PSO.C_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Liu, B., Wang, L., Jin, Y.H., Tang, F. and Huang, D.X., 2005. Improved particle swarm optimization
    combined with chaos. Chaos, Solitons & Fractals, 25(5), pp.1261-1271.
    """

    cdef public double c1
    cdef public double c2
    cdef public double w_max
    cdef public double w_min

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        w_min: float = 0.4,
        w_max: float = 0.9,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient, default = 2.05
            c2: [0-2] global coefficient, default = 2.05
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=PSOPopulation)
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.w_min = cy.validator(float, w_min, (0, 0.5), "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")
        self.parameters = ["epoch", "pop_size", "c1", "c2", "w_min", "w_max"]
        self.sort_flag = False

    def initialize_variables(self):
        pop_size = self.population.size()
        self.v_max = 0.5 * (self.problem.bounds.up - self.problem.bounds.low)
        self.v_min = -self.v_max
        self.N_CLS = int(pop_size / 5)  # Number of chaotic local searches
        self.dyn_lb = self.problem.bounds.low.copy()
        self.dyn_ub = self.problem.bounds.up.copy()

    def get_weights__(self, fit, fit_avg, fit_min):
        temp1 = self.w_min + (self.w_max - self.w_min) * (fit - fit_min) / (
            fit_avg - fit_min
        )
        if self.problem.sense == "min":
            output = temp1 if fit <= fit_avg else self.w_max
        else:
            output = self.w_max if fit <= fit_avg else temp1
        return output

    def bounded_solution(self, solution: np.ndarray) -> np.ndarray:
        return np.clip(solution, self.dyn_lb, self.dyn_ub)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        list_fits = [agent.fitness for agent in self.population]
        fit_avg = np.mean(list_fits)
        fit_min = np.min(list_fits)
        for idx in range(pop_size):
            w = self.get_weights__(self.population[idx].fitness, fit_avg, fit_min)
            v_new = (
                w * self.population[idx].velocity
                + self.c1
                * self.generator.random()
                * (self.population[idx].pbest_solution - self.population[idx].solution)
                + self.c2
                * self.generator.random()
                * (self.g_best.solution - self.population[idx].solution)
            )
            v_new = np.clip(v_new, self.v_min, self.v_max)
            x_new = self.population[idx].solution + v_new
            self.population[idx].velocity = v_new
            pos_new = self.bounded_solution(x_new)
            pos_new = self.population.correct_solution(pos_new)
            candidate = self.population.evaluate_solution(pos_new)
            if cy.is_better(candidate, self.population[idx], self.problem.sense):
                self.population[idx].update_solution(candidate)
            self.population[idx].update_pbest(candidate, self.problem.sense)

        ## Implement chaostic local search for the best solution
        g_best = self.g_best.copy()
        cx_best_0 = (self.g_best.solution - self.problem.bounds.low) / (
            self.problem.bounds.up - self.problem.bounds.low
        )  # Eq. 7
        cx_best_1 = 4 * cx_best_0 * (1 - cx_best_0)  # Eq. 6
        x_best = self.problem.bounds.low + cx_best_1 * (
            self.problem.bounds.up - self.problem.bounds.low
        )  # Eq. 8
        x_best = self.population.correct_solution(x_best)
        target_best = self.population.evaluate_solution(x_best)
        if cy.is_better(target_best, self.g_best, "min"):
            g_best.update_solution(target_best, x_best)

        r = self.generator.random()
        bound_min = np.stack(
            [self.dyn_lb, g_best.solution - r * (self.dyn_ub - self.dyn_lb)]
        )
        self.dyn_lb = np.max(bound_min, axis=0)
        bound_max = np.stack(
            [self.dyn_ub, g_best.solution + r * (self.dyn_ub - self.dyn_lb)]
        )
        self.dyn_ub = np.min(bound_max, axis=0)

        pop_new_child = self.population.generate(pop_size - self.N_CLS)
        self.population = cy.sort_agents(self.population + pop_new_child, self.problem.sense)[:pop_size]
