#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class CleverBookBeesA(VectorizeOptimizer):
    """
    The original version of: Bees Algorithm (CB-BeesA)

    Notes:
        + This version is based on ABC in the book Clever Algorithms
        + Improved the function search_neighborhood__

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_elites (int): number of employed bees which provided for good location
        + n_others (int): number of employed bees which provided for other location
        + patch_size (float): patch_variables = patch_variables * patch_reduction
        + patch_reduction (float): the reduction factor
        + n_sites (int): 3 bees (employed bees, onlookers and scouts),
        + n_elite_sites (int): 1 good partition

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import BeesA    >>> import numpy as np
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
    >>> model = BeesA.CleverBookBeesA(epoch=1000, pop_size=50, n_elites = 16, n_others = 4,
    >>>             patch_size = 5.0, patch_reduction = 0.985, n_sites = 3, n_elite_sites = 1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] D. T. Pham, Ghanbarzadeh A., Koc E., Otri S., Rahim S., and M.Zaidi. The bees algorithm - a novel tool
    for complex optimisation problems. In Proceedings of IPROMS 2006 Conference, pages 454–461, 2006.
    """

    cdef public object n_elites
    cdef public object n_others
    cdef public object patch_size
    cdef public object patch_reduction
    cdef public object n_sites
    cdef public object n_elite_sites

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_elites: int = 16,
        n_others: int = 4,
        patch_size: float = 5.0,
        patch_reduction: float = 0.985,
        n_sites: int = 3,
        n_elite_sites: int = 1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_elites (int): number of employed bees which provided for good location
            n_others (int): number of employed bees which provided for other location
            patch_size (float): patch_variables = patch_variables * patch_reduction
            patch_reduction (float): the reduction factor
            n_sites (int): 3 bees (employed bees, onlookers and scouts),
            n_elite_sites (int): 1 good partition
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "n_elites",
                "n_others",
                "patch_size",
                "patch_reduction",
                "n_sites",
                "n_elite_sites",
            ],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.n_elites = cy.validator(int, n_elites, [4, 20], "n_elites")
        self.n_others = cy.validator(int, n_others, [2, 5], "n_others")
        self.patch_size = cy.validator(float, patch_size, [2, 10], "patch_size")
        self.patch_reduction = cy.validator(float, patch_reduction, (0, 1.0), "patch_reduction")
        self.n_sites = cy.validator(int, n_sites, [2, 5], "n_sites")
        self.n_elite_sites = cy.validator(int, n_elite_sites, [1, 3], "n_elite_sites")

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, fresh
        cdef Py_ssize_t n = pop.n, d = pop.d, ns = self.n_sites
        cdef object rng = self.generator
        # sites: elite sites recruit n_elites bees, the other sites n_others; each bee changes one coordinate
        counts = np.zeros(n, dtype=int)
        counts[:ns] = self.n_others
        counts[:self.n_elite_sites] = self.n_elites
        parent = np.repeat(np.arange(n), counts)
        m = len(parent)
        pos = np.array(pop.X[parent])
        shift = rng.uniform(size=m) * self.patch_size * np.where(rng.uniform(size=m) < 0.5, 1.0, -1.0)
        pos[np.arange(m), rng.integers(0, d - 1, size=m)] += shift
        cand = pop.take(parent)
        cand.X[:] = self._correct_solution(pos)
        self.evaluate(cand, 0, m)
        ops.scatter(self, cand, parent)
        # the other bees scout random sources, kept only if they beat the bee they replace
        if ns < n:
            fresh = pop.take(np.arange(ns, n))
            fresh.X[:] = self.problem.bounds.low + rng.random((n - ns, d)) * (self.problem.bounds.up - self.problem.bounds.low)
            self.evaluate(fresh, 0, n - ns)
            ops.scatter(self, fresh, np.arange(ns, n))
