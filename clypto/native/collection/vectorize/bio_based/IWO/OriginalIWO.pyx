#!/usr/bin/env python
# Created by "Thieu" at 12:17, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalIWO(LegacyNativeOptimizer):
    """
    The original version of: Invasive Weed Optimization (IWO)

    Links:
        1. https://pdfs.semanticscholar.org/734c/66e3757620d3d4016410057ee92f72a9853d.pdf

    Notes:
        Better to use normal distribution instead of uniform distribution,
        updating population by sorting both parent population and child population

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + seed_min (int): [1, 3], Number of Seeds (min)
        + seed_max (int): [4, pop_size/2], Number of Seeds (max)
        + exponent (int): [2, 4], Variance Reduction Exponent
        + sigma_start (float): [0.5, 5.0], The initial value of Standard Deviation
        + sigma_end (float): (0, 0.5), The final value of Standard Deviation

    Examples
    ~~~~~~~~
    >>> from clypto.collection.bio_based import IWO    >>> import numpy as np
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
    >>> model = IWO.OriginalIWO(epoch=1000, pop_size=50, seed_min = 3, seed_max = 9, exponent = 3, sigma_start = 0.6, sigma_end = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Mehrabian, A.R. and Lucas, C., 2006. A novel numerical optimization algorithm inspired from weed colonization.
    Ecological informatics, 1(4), pp.355-366.
    """

    cdef public object seed_min
    cdef public object seed_max
    cdef public object exponent
    cdef public object sigma_start
    cdef public object sigma_end

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        seed_min: int = 2,
        seed_max: int = 10,
        exponent: int = 2,
        sigma_start: float = 1.0,
        sigma_end: float = 0.01,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            seed_min (int): Number of Seeds (min)
            seed_max (int): Number of seeds (max)
            exponent (int): Variance Reduction Exponent
            sigma_start (float): The initial value of standard deviation
            sigma_end (float): The final value of standard deviation
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "seed_min",
                "seed_max",
                "exponent",
                "sigma_start",
                "sigma_end",
            ],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.seed_min = cy.validator(int, seed_min, [1, 3], "seed_min")
        self.seed_max = cy.validator(int, seed_max, [4, int(self.pop_size / 2)], "seed_max")
        self.exponent = cy.validator(int, exponent, [2, 4], "exponent")
        self.sigma_start = cy.validator(float, sigma_start, [0.5, 5.0], "sigma_start")
        self.sigma_end = cy.validator(float, sigma_end, (0, 0.5), "sigma_end")

    cdef void evolve(self, int epoch_c):
        # The number of seeds of each plant depends on its fitness ratio: candidates are built
        # plant by plant (same draw order), evaluated together and the best ones survive.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t idx, jdx
        # Update Standard Deviation
        sigma = (1.0 - epoch_c / self.epoch) ** self.exponent * (self.sigma_start - self.sigma_end) + self.sigma_end
        order = self.sorted_order(pop)
        Xs, Fs = pop.X[order], pop.F[order]  # the sorted population
        best_fit, worst_fit = Fs[0], Fs[-1]
        moves = []
        for idx in range(0, self.pop_size):
            temp = best_fit - worst_fit
            if temp == 0:
                ratio = self.generator.random()
            else:
                ratio = (Fs[idx] - worst_fit) / temp
            s = int(np.ceil(self.seed_min + (self.seed_max - self.seed_min) * ratio))
            if s > int(np.sqrt(self.pop_size)):
                s = int(np.sqrt(self.pop_size))
            for jdx in range(s):
                # Initialize Offspring and Generate Random Location
                pos_new = Xs[idx] + sigma * self.generator.normal(0, 1, self.problem.n_dims)
                moves.append(self.correct_solution(pos_new))
        cand = self.new_population(np.array(moves))
        self.pop = cand.take(self.sorted_order(cand)[:self.pop_size])
