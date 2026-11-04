#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from scipy.stats import cauchy
cimport clypto.core as cy



cdef class JADE(cy.Optimizer):
    """
    The original version of: Differential Evolution (JADE)

    Links:
        1. https://doi.org/10.1109/TEVC.2009.2014613

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial adaptive f, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial adaptive cr, default = 0.5
        + pt (float): [0.05, 0.2], The percent of top best agents (p in the paper), default = 0.1
        + ap (float): [0.05, 0.2], The Adaptation Parameter control value of f and cr (c in the paper), default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.JADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional
    external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958.
    """

    cdef public double ap
    cdef public double miu_cr
    cdef public double miu_f
    cdef public double pt

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        miu_f: float = 0.5,
        miu_cr: float = 0.5,
        pt: float = 0.1,
        ap: float = 0.1,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial adaptive f, default = 0.5
            miu_cr (float): initial adaptive cr, default = 0.5
            pt (float): The percent of top best agents (p in the paper), default = 0.1
            ap (float): The Adaptation Parameter control value of f and cr (c in the paper), default=0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "miu_f", "miu_cr", "pt", "ap"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.miu_f = cy.validator(float, miu_f, (0, 1.0), "miu_f")
        self.miu_cr = cy.validator(float, miu_cr, (0, 1.0), "miu_cr")
        # np.random.uniform(0.05, 0.2) # the x_best is select from the top 100p % solutions
        self.pt = cy.validator(float, pt, (0, 1.0), "pt")
        # np.random.uniform(1/20, 1/5) # the adaptation parameter control value of f and cr
        self.ap = cy.validator(float, ap, (0, 1.0), "ap")

    def initialize_variables(self):
        self.dyn_miu_cr = self.miu_cr
        self.dyn_miu_f = self.miu_f
        self.dyn_pop_archive = list()

    ### Survivor Selection
    def lehmer_mean(self, list_objects):
        temp = np.sum(list_objects)
        return 0 if temp == 0 else np.sum(list_objects**2) / temp

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        list_f = list()
        list_cr = list()
        temp_f = list()
        temp_cr = list()
        pop_sorted = self.population.sort()
        pop = []
        for idx in range(0, pop_size):
            ## Calculate adaptive parameter cr and f
            cr = self.generator.normal(self.dyn_miu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f, 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            temp_f.append(f)
            temp_cr.append(cr)
            top = int(pop_size * self.pt)
            x_best = pop_sorted[self.generator.integers(0, top)]
            r1_idx = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            new_pop = self.population + self.dyn_pop_archive
            r2_idx = self.generator.choice(
                list(set(range(0, len(new_pop))) - {idx, r1_idx})
            )
            x_r1 = self.population[r1_idx].solution
            x_r2 = new_pop[r2_idx].solution
            x_new = (
                self.population[idx].solution
                + f * (x_best.solution - self.population[idx].solution)
                + f * (x_r1 - x_r2)
            )
            x = np.where(
                self.generator.random(self.problem.n_dims) < cr,
                x_new,
                self.population[idx].solution,
            )
            j_rand = self.generator.integers(0, self.problem.n_dims)
            x[j_rand] = x_new[j_rand]
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop.append(agent)
        pop = self.population.evaluate(pop, self.mode)
        for idx, agent in enumerate(self.population.toarray()):
            if cy.is_better(pop[idx], agent, self.problem.sense):
                self.dyn_pop_archive.append(cy.duplicate_agent(agent))
                list_cr.append(temp_cr[idx])
                list_f.append(temp_f[idx])
                self.population[idx] = cy.duplicate_agent(pop[idx])
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - pop_size
        if temp > 0:
            idx_list = self.generator.choice(
                range(0, len(self.dyn_pop_archive)), temp, replace=False
            )
            archive_pop_new = []
            for idx, solution in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(solution)
            self.dyn_pop_archive = archive_pop_new
        # Update miu_cr and miu_f
        if len(list_cr) == 0:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * 0.5
        else:
            self.dyn_miu_cr = (1 - self.ap) * self.dyn_miu_cr + self.ap * np.mean(
                np.array(list_cr)
            )
        if len(list_f) == 0:
            self.dyn_miu_f = (1 - self.ap) * self.dyn_miu_f + self.ap * 0.5
        else:
            self.dyn_miu_f = (
                1 - self.ap
            ) * self.dyn_miu_f + self.ap * self.lehmer_mean(np.array(list_f))
