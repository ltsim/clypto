#!/usr/bin/env python
# Created by "Thieu" at 19:38, 10/03/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.stats import qmc
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


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
    >>> from clypto.native.collection.vectorize.math_based import PSS    >>> import numpy as np
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
        n = self.pop_size if pop_size is None else pop_size
        if self.sampling_method == "MC":
            return self.generator.random((n, self.problem.n_dims))
        # default: "LHS" (the sampler draws from our generator, so a seed reproduces the run)
        return qmc.LatinHypercube(d=self.problem.n_dims, rng=self.generator).random(n=n)

    cdef void initialization(self):
        lb, ub = self.problem.lb, self.problem.ub
        pos = np.round((lb + self.create_population(self.pop_size) * (ub - lb)) / self.steps) * self.steps
        self.pop = self.new_population(self.correct_solution(pos))

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        lb, ub = self.problem.lb, self.problem.ub
        gb = self.current_g_best()
        g, g_fit = np.array(gb.solution), gb.target.fitness
        rand = self.create_population(n)
        if self.new_solution:
            deviation = np.abs(0.5 * (1.0 - self.acceptance_rate) * (ub - lb)) * (1 - (epoch_c / self.epoch))
        else:
            deviation = rng.uniform(np.minimum(0, g), np.maximum(0, g), (n, d))
        reduced_lb = np.maximum(g - deviation, lb)
        reduced_ub = np.minimum(reduced_lb + deviation * 2.0, ub)
        pos = np.where(rng.random((n, d)) <= self.acceptance_rate,
                       reduced_lb + rand * (reduced_ub - reduced_lb), lb + rand * (ub - lb))
        ops.replace(self, np.round(pos / self.steps) * self.steps)
        self.new_solution = bool(ops.better(self, np.asarray(self.pop.F)[ops.best_row(self, self.pop)], g_fit))
