#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class ImprovedNMRA(LegacyNativeOptimizer):
    """
    The developed version of: Improved Naked Mole-Rat Algorithm (I-NMRA)

    Notes:
        + Use mutation probability idea
        + Use crossover operator
        + Use Levy-flight technique

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.5, 0.95], probability of breeding, default = 0.75
        + pm (float): [0.01, 0.1], probability of mutation, default = 0.01

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import NMRA    >>> import numpy as np
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
    >>> model = NMRA.ImprovedNMRA(epoch=1000, pop_size=50, pb = 0.75, pm = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object pb
    cdef public object pm
    cdef public object size_b

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        pb = 0.75,
        pm = 0.01,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pb (float): breeding probability, default = 0.75
            pm (float): probability of mutation, default = 0.01
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pb", "pm"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.size_b = int(self.pop_size / 5)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d, b = self.size_b
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        me = np.arange(n)
        # breeding operators (the first size_b agents)
        levy = self.get_levy_flight_step(beta=1, multiplier=0.001, size=(n, 1), case=-1)
        breed = np.where((rng.uniform(size=n) < self.pb)[:, None], X + rng.normal(0, 1, (n, d)) * (g - X),
                         X + 1.0 / np.sqrt(epoch_c) * np.sign(rng.random((n, 1)) - 0.5) * levy * (X - g))
        # working operators: a difference of two workers, or a crossover of the best with a random partner
        m = n - b
        t1 = b + rng.integers(0, m, size=n)
        t2 = b + (t1 - b + rng.integers(1, m, size=n)) % m
        start = rng.integers(0, d / 2, size=(n, 1))
        cols = np.arange(d)[None, :]
        inside = (cols >= start) & (cols < (start + int(d / 3)))
        cross = np.where(inside, X[rng.integers(0, n, size=n)], g)
        work = np.where((rng.uniform(size=n) < 0.5)[:, None], X + rng.normal(0, 1, (n, d)) * (X[t1] - X[t2]), cross)
        pos = np.where((me < b)[:, None], breed, work)
        pos = np.where(rng.uniform(0, 1, (n, d)) < self.pm, rng.uniform(lb, ub, (n, d)), pos)  # mutation
        ops.step(self, pos)
