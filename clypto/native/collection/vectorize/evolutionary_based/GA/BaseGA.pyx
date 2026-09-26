#!/usr/bin/env python
# Created by "Thieu" at 09:33, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class BaseGA(VectorizeOptimizer):
    """
    The original version of: Genetic Algorithm (GA)

    Links:
        1. https://blog.sicara.com/getting-started-genetic-algorithms-python-tutorial-81ffa1dd72f9
        2. https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm
        3. https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.95
        + pm (float): [0.01, 0.2], mutation probability, default = 0.025
        + selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
        + k_way (float): Optional, set it when use "tournament" selection, default = 0.2
        + crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
        + mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = True
        + mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.evolutionary_based import GA    >>> import numpy as np
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
    >>> model = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    >>>
    >>> model2 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="tournament", k_way=0.4, crossover="multi_points")
    >>>
    >>> model3 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="one_point", mutation="scramble")
    >>>
    >>> model4 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation_multipoints=True, mutation="swap")
    >>>
    >>> model5 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="roulette", crossover="multi_points")
    >>>
    >>> model6 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, selection="random", mutation="inversion")
    >>>
    >>> model7 = GA.BaseGA(epoch=1000, pop_size=50, pc=0.9, pm=0.05, crossover="arithmetic", mutation="flip")

    References
    ~~~~~~~~~~
    [1] Whitley, D., 1994. A genetic algorithm tutorial. Statistics and computing, 4(2), pp.65-85.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pc: float = 0.95,
        pm: float = 0.025,
        *,
        selection: str | None = None,
        k_way: float | None = None,
        crossover: str | None = None,
        mutation_multipoints: bool | None = None,
        mutation: str | None = None,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            pc: cross-over probability, default = 0.95
            pm: mutation probability, default = 0.025
            selection (str): Optional, can be ["roulette", "tournament", "random"], default = "tournament"
            k_way (float): Optional, set it when use "tournament" selection, default = 0.2
            crossover (str): Optional, can be ["one_point", "multi_points", "uniform", "arithmetic"], default = "uniform"
            mutation_multipoints (bool): Optional, True or False, effect on mutation process, default = False
            mutation (str): Optional, can be ["flip", "swap"] for multipoints and can be ["flip", "swap", "scramble", "inversion"] for one-point, default="flip"
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pc", "pm"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pc = cy.validator(float, pc, (0, 1.0), "pc")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.selection = "tournament"
        self.k_way = 0.2
        self.crossover = "uniform"
        self.mutation = "flip"
        self.mutation_multipoints = True

        if selection is not None:
            self.selection = cy.validator(str, selection, ["tournament", "random", "roulette"], "selection")
        if k_way is not None:
            self.k_way = cy.validator(float, k_way, (0, 1.0), "k_way")
        if crossover is not None:
            self.crossover = cy.validator(str, crossover, ["one_point", "multi_points", "uniform", "arithmetic"], "crossover")
        if mutation_multipoints is not None:
            self.mutation_multipoints = cy.validator(bool, mutation_multipoints, None, "mutation_multipoints")
        if mutation is not None:
            if self.mutation_multipoints:
                self.mutation = cy.validator(str, mutation, ["flip", "swap"], "mutation")
            else:
                self.mutation = cy.validator(str, mutation, ["flip", "swap", "scramble", "inversion"], "mutation")

    def select_pairs__(self, F, m):
        """``m`` pairs of parents (two index arrays into the mating pool with fitness ``F``)."""
        rng = self.generator
        N = len(F)
        if self.selection == "roulette":
            i1, i2 = ops.roulette(self, F, m), ops.roulette(self, F, m)
            i2 = np.where(i1 == i2, (i1 + rng.integers(1, N, size=m)) % N, i2)
        elif self.selection == "random":
            i1 = rng.integers(0, N, size=m)
            i2 = (i1 + rng.integers(1, N, size=m)) % N
        else:  # tournament
            top = self.tournament__(F, m, 2)
            i1, i2 = top[:, 0], top[:, 1]
        return i1, i2

    def select_pairs_pools__(self, F_dad, F_mom, m):
        """One parent from each of two mating pools."""
        rng = self.generator
        if self.selection == "roulette":
            return ops.roulette(self, F_dad, m), ops.roulette(self, F_mom, m)
        if self.selection == "random":
            return rng.integers(0, len(F_dad), size=m), rng.integers(0, len(F_mom), size=m)
        return self.tournament__(F_dad, m, 1)[:, 0], self.tournament__(F_mom, m, 1)[:, 0]

    def tournament__(self, F, m, output, reverse=False):
        """``m`` k-way tournaments: the ``output`` best (worst when ``reverse``) of ``k`` random agents each, (m, output)."""
        F = np.asarray(F)
        N = len(F)
        k = int(self.k_way * N) if 0 < self.k_way < 1 else int(self.k_way)
        k = max(k, output)
        picked = np.argpartition(self.generator.random((m, N)), k - 1, axis=1)[:, :k]
        order = np.argsort(F[picked], axis=1)
        if (self.problem.sense == "max") != bool(reverse):
            order = order[:, ::-1]
        return np.take_along_axis(picked, order[:, :output], axis=1)

    def crossover_batch__(self, dad, mom):
        rng = self.generator
        m, d = dad.shape
        cols = np.arange(d)[None, :]
        if self.crossover == "arithmetic":
            r = rng.uniform(size=(m, 1))
            return r * dad + (1 - r) * mom, r * mom + (1 - r) * dad
        if self.crossover == "one_point":
            mask = cols < rng.integers(1, d - 1, size=(m, 1))
            return np.where(mask, dad, mom), np.where(mask, mom, dad)
        if self.crossover == "multi_points":
            a = rng.integers(1, d - 1, size=(m, 1))
            b = (a - 1 + rng.integers(1, d - 2, size=(m, 1))) % (d - 2) + 1  # a second, different cut
            mask = (cols >= np.minimum(a, b)) & (cols < np.maximum(a, b))
            return np.where(mask, mom, dad), np.where(mask, dad, mom)
        flip = rng.integers(0, 2, size=(m, d))  # uniform
        return dad * flip + mom * (1 - flip), mom * flip + dad * (1 - flip)

    def mutation_batch__(self, child, multipoints=None):
        rng = self.generator
        m, d = child.shape
        me = np.arange(m)
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        child = np.array(child)
        if self.mutation_multipoints if multipoints is None else multipoints:
            if self.mutation == "swap":  # (the classic loop returns after its first swap: gene 0 with a random gene)
                j = rng.integers(1, d, size=m)
                tmp = child[me, 0].copy()
                child[me, 0], child[me, j] = child[me, j], tmp
                return child
            return np.where(rng.uniform(0, 1, (m, d)) < self.pm, rng.uniform(lb, ub, (m, d)), child)  # flip
        if self.mutation == "swap":
            i1 = rng.integers(0, d, size=m)
            i2 = (i1 + rng.integers(1, d, size=m)) % d
            tmp = child[me, i1].copy()
            child[me, i1], child[me, i2] = child[me, i2], tmp
        elif self.mutation in ("inversion", "scramble"):
            for i in range(m):
                cut1, cut2 = rng.choice(d, 2, replace=False)
                seg = child[i, cut1:cut2]
                child[i, cut1:cut2] = seg[::-1] if self.mutation == "inversion" else rng.permutation(seg)
        else:  # "flip"
            j = rng.integers(0, d, size=m)
            child[me, j] = lb[j] + rng.random(m) * (ub[j] - lb[j])
        return child

    def breed__(self, pool_x, i1, i2):
        """Children of the parent pairs ``(i1, i2)``: crossover (probability pc) then mutation; two children per pair."""
        dad, mom = pool_x[i1], pool_x[i2]
        cross = (self.generator.random(len(i1)) < self.pc)[:, None]
        c1, c2 = self.crossover_batch__(dad, mom)
        c1, c2 = np.where(cross, c1, dad), np.where(cross, c2, mom)
        return self.mutation_batch__(np.vstack([c1, c2]))

    def elite_step__(self):
        """Elite strategies: the best agents survive, the rest is replaced by children (one child per pair)."""
        cdef NativePopulation pop = self.pop
        cdef NativePopulation kids
        cdef Py_ssize_t n = pop.n, e = self.n_elite_best
        X, F = np.asarray(pop.X), np.asarray(pop.F)
        m = n - e
        if self.strategy == 0:
            i1, i2 = self.select_pairs__(F[e:], m)
            pool = X[e:]
            dad, mom = pool[i1], pool[i2]
        else:
            w = self.n_elite_worst
            i1, i2 = self.select_pairs_pools__(F[e:e + w], F[e + w:], m)
            dad, mom = X[e:e + w][i1], X[e + w:][i2]
        cross = (self.generator.random(m) < self.pc)[:, None]
        c1, c2 = self.crossover_batch__(dad, mom)
        c1, c2 = np.where(cross, c1, dad), np.where(cross, c2, mom)
        child = np.where((self.generator.random(m) <= 0.5)[:, None], c1, c2)
        kids = pop.take(np.arange(e, n))
        kids.X[:] = self._correct_solution(self.mutation_batch__(child))
        self.evaluate(kids, 0, m)
        self.pop = pop.take(np.arange(e)).concat(kids)

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation kids, both
        cdef Py_ssize_t n = pop.n
        X, F = np.asarray(pop.X), np.asarray(pop.F)
        i1, i2 = self.select_pairs__(F, -(-n // 2))  # ceil division, safe for odd pop_size
        children = self.breed__(X, i1, i2)
        kids = pop.take(np.arange(n))
        kids.X[:] = self._correct_solution(children[:n])
        self.evaluate(kids, 0, n)
        # survivor selection: every child fights the worst of a random tenth of the population
        rival = self.tournament__(F, n, 1, reverse=True)[:, 0]
        wins = ops.better(self, np.asarray(kids.F), F[rival])
        both = pop.concat(kids)
        self.pop = both.take(np.where(wins, n + np.arange(n), rival))
