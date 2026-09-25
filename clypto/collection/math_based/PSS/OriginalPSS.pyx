#!/usr/bin/env python
# Created by "Thieu" at 19:38, 10/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.stats import qmc
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalPSS(LegacyNativeOptimizer):
    """
    The original version of: Pareto-like Sequential Sampling (PSS)

    Links:
        1. https://doi.org/10.1007/s00500-021-05853-8
        2. https://github.com/eesd-epfl/pareto-optimizer

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + acceptance_rate (float): [0.7-0.96], the probability of accepting a solution in the normal range, default=0.9
        + sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default="LHS"

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import PSS    >>> import numpy as np
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
    >>> model = PSS.OriginalPSS(epoch=1000, pop_size=50, acceptance_rate = 0.8, sampling_method = "LHS")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shaqfa, M. and Beyer, K., 2021. Pareto-like sequential sampling heuristic for global optimisation. Soft Computing, 25(14), pp.9077-9096.
    """

    cdef public double acceptance_rate
    cdef public object sampling_method
    cdef public object step
    cdef public object steps
    cdef public object new_solution

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        acceptance_rate: float = 0.9,
        sampling_method: str = "LHS",
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            acceptance_rate (float): the probability of accepting a solution in the normal range, default = 0.9
            sampling_method (str): 'LHS': Latin-Hypercube or 'MC': 'MonteCarlo', default = "LHS"
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "acceptance_rate", "sampling_method"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.acceptance_rate = cy.validator(float, acceptance_rate, (0, 1.0), "acceptance_rate")
        self.sampling_method = cy.validator(str, sampling_method, ["MC", "LHS"], "sampling_method")

    cdef void initialize_variables(self):
        self.step = 10e-10
        self.steps = np.ones(self.problem.n_dims) * self.step
        self.new_solution = True

    def create_population(self, pop_size=None):
        if self.sampling_method == "MC":
            pop = self.generator.random(self.pop_size, self.problem.n_dims)
        else:  # Default: "LHS"
            sampler = qmc.LatinHypercube(d=self.problem.n_dims)
            pop = sampler.random(n=pop_size)
        return pop

    cdef void initialization(self):
        lb_pop = np.repeat(np.reshape(self.problem.lb, (1, -1)), self.pop_size, axis=0)
        ub_pop = np.repeat(np.reshape(self.problem.ub, (1, -1)), self.pop_size, axis=0)
        steps_mat = np.repeat(np.reshape(self.steps, (1, -1)), self.pop_size, axis=0)
        random_pop = self.create_population(self.pop_size)
        pop = (
                np.round((lb_pop + random_pop * (ub_pop - lb_pop)) / steps_mat) * steps_mat
        )
        self.pop = self.new_population(np.array([self.correct_solution(pos) for pos in pop]))

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, k, n = self.pop_size, d = pop.d
        lb, ub = self.problem.lb, self.problem.ub
        gb = self.current_g_best()
        g_best_pos, g_best_fit = gb.solution, gb.target.fitness
        pop_rand = self.create_population(self.pop_size)
        Xp, Xc = pop.X, cand.X
        for idx in range(0, n):
            pos_new = Xp[idx].copy()
            for k in range(d):
                # Update the ranges
                deviation = self.generator.uniform(min(0, g_best_pos[k]), max(0, g_best_pos[k]))
                if self.new_solution:
                    # The deviation is positive dynamic real number
                    deviation = abs(0.5 * (1.0 - self.acceptance_rate) * (ub[k] - lb[k])) * (1 - (epoch / self.epoch))
                reduced_lb = g_best_pos[k] - deviation
                reduced_lb = np.amax([reduced_lb, lb[k]])
                reduced_ub = reduced_lb + deviation * 2.0
                reduced_ub = np.amin([reduced_ub, ub[k]])
                # Choose new solution
                if self.generator.random() <= self.acceptance_rate:
                    # choose a solution from the prominent domain
                    pos_new[k] = reduced_lb + pop_rand[idx, k] * (reduced_ub - reduced_lb)
                else:
                    # choose a solution from the overall domain
                    pos_new[k] = lb[k] + pop_rand[idx, k] * (ub[k] - lb[k])
                # Round for the step size
                pos_new = np.round(pos_new / self.steps) * self.steps
            # Check the bound
            Xc[idx] = self.correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand
        best = self.sorted_order(cand)[0]
        if self.compare_fitness(cand.F[best], g_best_fit, self.problem.minmax):
            self.new_solution = True
        else:
            self.new_solution = False
