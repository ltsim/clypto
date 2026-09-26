#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalEOA(VectorizeOptimizer):
    """
    The developed version: Earthworm Optimisation Algorithm (EOA)

    Links:
        1. http://doi.org/10.1504/IJBIC.2015.10004283
        2. https://www.mathworks.com/matlabcentral/fileexchange/53479-earthworm-optimization-algorithm-ewa

    Notes:
        The original version from matlab code above will not work well, even with small dimensions.
        I change updating process, change cauchy process using x_mean, use global best solution

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + p_c (float): (0, 1) -> better [0.5, 0.95], crossover probability
        + p_m (float): (0, 1) -> better [0.01, 0.2], initial mutation probability
        + n_best (int): (2, pop_size/2) -> better [2, 5], how many of the best earthworm to keep from one generation to the next
        + alpha (float): (0, 1) -> better [0.8, 0.99], similarity factor
        + beta (float): (0, 1) -> better [0.8, 1.0], the initial proportional factor
        + gama (float): (0, 1) -> better [0.8, 0.99], a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import EOA    >>> import numpy as np
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
    >>> model = EOA.OriginalEOA(epoch=1000, pop_size=50, p_c = 0.9, p_m = 0.01, n_best = 2, alpha = 0.98, beta = 0.9, gama = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Wang, G.G., Deb, S. and Coelho, L.D.S., 2018. Earthworm optimisation algorithm: a bio-inspired metaheuristic algorithm
    for global optimisation problems. International journal of bio-inspired computation, 12(1), pp.1-22.
    """

    cdef public object p_c
    cdef public object p_m
    cdef public object n_best
    cdef public object alpha
    cdef public object beta
    cdef public object gama
    cdef public object dyn_beta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        p_c: float = 0.9,
        p_m: float = 0.01,
        n_best: int = 2,
        alpha: float = 0.98,
        beta: float = 0.9,
        gama: float = 0.9,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            p_c (float): default = 0.9, crossover probability
            p_m (float): default = 0.01 initial mutation probability
            n_best (int): default = 2, how many of the best earthworm to keep from one generation to the next
            alpha (float): default = 0.98, similarity factor
            beta (float): default = 0.9, the initial proportional factor
            gama (float): default = 0.9, a constant that is similar to cooling factor of a cooling schedule in the simulated annealing.
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "p_c", "p_m", "n_best", "alpha", "beta", "gama"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.p_c = cy.validator(float, p_c, (0, 1.0), "p_c")
        self.p_m = cy.validator(float, p_m, (0, 1.0), "p_m")
        self.n_best = cy.validator(int, n_best, [2, int(self.pop_size / 2)], "n_best")
        self.alpha = cy.validator(float, alpha, (0, 1.0), "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")
        self.gama = cy.validator(float, gama, (0, 1.0), "gama")

    def _initialize_variables(self):
        self.dyn_beta = self.beta

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativePopulation elites, merged
        cdef NativeTarget tar
        cdef Py_ssize_t i, idx, n = pop.n
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        sense = self.problem.sense
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        ## Update the pop best
        elites = pop.take(self.sorted_order(pop)[:self.n_best])
        for i in range(0, self.pop_size):
            idx = i
            ### Reproduction 1: the first way of reproducing
            x_t1 = self.problem.bounds.low + self.problem.bounds.up - self.alpha * Xp[idx]

            ### Reproduction 2: the second way of reproducing
            if idx >= self.n_best:  ### Select two parents to mate and create two children
                idx = int(self.pop_size * 0.2)  # (the classic code re-uses the agent index here)
                if self.generator.uniform() < 0.5:  ## 80% parents selected from best population
                    idx1, idx2 = self.generator.choice(range(0, idx), 2, replace=False)
                else:  ## 20% left parents selected from worst population (make more diversity)
                    idx1, idx2 = self.generator.choice(range(idx, self.pop_size), 2, replace=False)
                r = self.generator.uniform()
                x_child = r * Xp[idx2] + (1 - r) * Xp[idx1]
            else:
                r1 = self.generator.integers(0, self.pop_size)
                x_child = Xp[r1]
            x_t1 = self.dyn_beta * x_t1 + (1.0 - self.dyn_beta) * x_child
            # sequential mode replaces row idx (possibly the re-used index); swarm modes keep agent i's candidate
            ops.commit(self, pop, cand, i if swarm else idx, self._correct_solution(x_t1), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
        self.dyn_beta = self.gama * self.beta
        pop = pop.take(self.sorted_order(pop)[:self.pop_size])
        self.pop = pop
        Xp = pop.X

        pos_list = np.array(pop.X)
        x_mean = np.mean(pos_list, axis=0)
        ## Cauchy mutation (CM)
        cauchy_w = g_best.copy()
        cand = pop.empty_like()
        for idx in range(self.n_best, self.pop_size):  # Don't allow the elites to be mutated
            condition = self.generator.random(self.problem.n_dims) < self.p_m
            cauchy_w = np.where(condition, x_mean, cauchy_w)
            x_t1 = (cauchy_w + g_best) / 2
            ops.commit(self, pop, cand, idx, self._correct_solution(x_t1), swarm)
        if swarm:
            # greedy_selection_population(pop_new, pop[n_best:]): the old agent stays only if strictly better
            self.evaluate(cand, self.n_best, n)
            old, new = pop.F[self.n_best:], cand.F[self.n_best:]
            keep = old < new if sense == "min" else old > new
            rows = self.n_best + np.flatnonzero(~keep)
            pop.buf[rows] = cand.buf[rows]

        ## Elitism Strategy: Replace the worst with the previous generation's elites.
        pop = pop.take(self.sorted_order(pop))
        for idx in range(0, self.n_best):
            pop.buf[self.pop_size - idx - 1] = elites.buf[idx]

        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx in range(pop.n):
            key = tuple(pop.X[idx].tolist())
            if key in new_set:
                x = self.problem.generate_solution(True)
                ops.set_row(pop, idx, x, self._get_target(x))
            else:
                new_set.add(key)
        self.pop = pop
