#!/usr/bin/env python
# Created by "Thieu" at 08:37, 17/06/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.stats import cauchy
cimport clypto.core as cy



cdef class L_SHADE(cy.Optimizer):
    """
    The original version of: Linear Population Size Reduction Success-History Adaptation Differential Evolution (LSHADE)

    Links:
        1. https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + miu_f (float): [0.4, 0.6], initial weighting factor, default = 0.5
        + miu_cr (float): [0.4, 0.6], initial cross-over probability, default = 0.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.evolutionary_based import SHADE    >>> import numpy as np
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
    >>> model = SHADE.L_SHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Tanabe, R. and Fukunaga, A.S., 2014, July. Improving the search performance of SHADE using
    linear population size reduction. In 2014 IEEE congress on evolutionary computation (CEC) (pp. 1658-1665). IEEE.
    """

    cdef public double miu_cr
    cdef public double miu_f

    def __init__(
            self,
            epoch: int = 750,
            pop_size: int = 100,
            miu_f: float = 0.5,
            miu_cr: float = 0.5,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            miu_f (float): initial weighting factor, default = 0.5
            miu_cr (float): initial cross-over probability, default = 0.5
        """
        super().__init__(parameters=["epoch", "pop_size", "miu_f", "miu_cr"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.miu_f = cy.validator(float, miu_f, (0, 1.0), "miu_f")
        self.miu_cr = cy.validator(float, miu_cr, (0, 1.0), "miu_cr")

    def initialize_variables(self):
        pop_size = self.population.size()
        # Dynamic variable
        self.dyn_miu_f = self.miu_f * np.ones(pop_size)  # list the initial f,
        self.dyn_miu_cr = self.miu_cr * np.ones(pop_size)  # list the initial cr,
        self.dyn_pop_archive = list()
        self.dyn_pop_size = pop_size
        self.k_counter = 0
        self.n_min = int(pop_size / 5)

    ### Survivor Selection
    def weighted_lehmer_mean(self, list_objects, list_weights):
        up = np.sum(list_weights * list_objects ** 2)
        down = np.sum(list_weights * list_objects)
        return up / down if down != 0 else 0.5

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        list_f = list()
        list_cr = list()
        list_f_index = list()
        list_cr_index = list()
        list_f_new = np.ones(pop_size)
        list_cr_new = np.ones(pop_size)
        pop_old = [agent.copy() for agent in self.population]
        pop_sorted = self.population.sort()
        pop = []
        for idx in range(0, pop_size):
            ## Calculate adaptive parameter cr and f
            idx_rand = self.generator.integers(0, pop_size)
            cr = self.generator.normal(self.dyn_miu_cr[idx_rand], 0.1)
            cr = np.clip(cr, 0, 1)
            while True:
                f = cauchy.rvs(self.dyn_miu_f[idx_rand], 0.1)
                if f < 0:
                    continue
                elif f > 1:
                    f = 1
                break
            list_cr_new[idx] = cr
            list_f_new[idx] = f
            p = self.generator.uniform(0.15, 0.2)
            top = int(np.ceil(self.dyn_pop_size * p))
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
            pos_new = np.where(
                self.generator.random(self.problem.n_dims) < cr,
                x_new,
                self.population[idx].solution,
            )
            j_rand = self.generator.integers(0, self.problem.n_dims)
            pos_new[j_rand] = x_new[j_rand]
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop[-1].evaluate(self.problem)
        pop = self.population.evaluate(pop, self.mode)
        for idx in range(0, pop_size):
            if cy.is_better(pop[idx], self.population[idx], self.problem.sense):
                list_cr.append(list_cr_new[idx])
                list_f.append(list_f_new[idx])
                list_f_index.append(idx)
                list_cr_index.append(idx)
                self.population[idx] = pop[idx].copy()
                self.dyn_pop_archive.append(self.population[idx].copy())
        # Randomly remove solution
        temp = len(self.dyn_pop_archive) - pop_size
        if temp > 0:
            idx_list = self.generator.choice(
                range(0, len(self.dyn_pop_archive)), temp, replace=False
            )
            archive_pop_new = []
            for idx, agent in enumerate(self.dyn_pop_archive):
                if idx not in idx_list:
                    archive_pop_new.append(agent.copy())
            self.dyn_pop_archive = archive_pop_new
        # Update miu_cr and miu_f
        if len(list_f) != 0 and len(list_cr) != 0:
            # Eq.13, 14, 10
            list_fit_old = np.ones(len(list_cr_index))
            list_fit_new = np.ones(len(list_cr_index))
            idx_increase = 0
            for idx in range(0, self.dyn_pop_size):
                if idx in list_cr_index:
                    list_fit_old[idx_increase] = pop_old[idx].fitness
                    list_fit_new[idx_increase] = self.population[idx].fitness
                    idx_increase += 1
            total_fit = np.sum(np.abs(list_fit_new - list_fit_old))
            list_weights = (
                0 if total_fit == 0 else np.abs(list_fit_new - list_fit_old) / total_fit
            )
            self.dyn_miu_cr[self.k_counter] = np.sum(list_weights * np.array(list_cr))
            self.dyn_miu_f[self.k_counter] = self.weighted_lehmer_mean(
                np.array(list_f), list_weights
            )
            self.k_counter += 1
            if self.k_counter >= self.dyn_pop_size:
                self.k_counter = 0
        # Linear Population Size Reduction
        self.dyn_pop_size = round(
            pop_size + epoch * ((self.n_min - pop_size) / self.epoch)
        )
