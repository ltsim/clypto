#!/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalABC(LegacyNativeOptimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=25

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ABC    >>> import numpy as np
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
    >>> model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """

    cdef public object n_limits
    cdef public object trials

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_limits: int = 25,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_limits"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.n_limits = cy.validator(int, n_limits, [1, 1000], "n_limits")

    cdef void initialize_variables(self):
        self.trials = np.zeros(self.pop_size)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, scouts
        cdef Py_ssize_t i, n = pop.n, d = pop.d
        cdef object rng = self.generator
        # Employed bees: a random other bee as guide
        X = pop.X
        before = np.array(pop.F)
        phi = rng.uniform(low=-1, high=1, size=(n, d))
        ops.step(self, X + phi * (X[ops.others(self, n)[:, 0]] - X))
        improved = ops.better(self, pop.F, before)
        self.trials = np.where(improved, 0, self.trials + 1)
        # Onlooker bees: roulette wheel on the employed fitness (Eq. of the classic code), guided by another bee
        fits = np.array(pop.F)
        if np.ptp(fits) == 0:
            selected = rng.integers(0, n, size=n)
        else:
            f = fits - fits.min() if np.any(fits < 0) else fits
            f = f.max() - f if self.problem.minmax == "min" else f
            selected = rng.choice(n, size=n, p=f / f.sum())
        X = pop.X
        guide = (selected + rng.integers(1, n, size=n)) % n
        phi = rng.uniform(low=-1, high=1, size=(n, d))
        cand = pop.take(selected)
        cand.X[:] = self.correct_solution(X[selected] + phi * (X[guide] - X[selected]))
        self.evaluate(cand, 0, n)
        for i in range(n):  # bees chosen several times keep their best candidate
            s = selected[i]
            if ops.better(self, cand.F[i], pop.F[s]):
                pop.buf[s] = cand.buf[i]
                self.trials[s] = 0
            else:
                self.trials[s] += 1
        # Scout bees: abandon the food sources whose trials exceed the limit
        abandoned = np.flatnonzero(self.trials >= self.n_limits)
        if len(abandoned):
            scouts = pop.take(abandoned)
            scouts.X[:] = self.problem.lb + rng.random((len(abandoned), d)) * (self.problem.ub - self.problem.lb)
            self.evaluate(scouts, 0, len(abandoned))
            pop.buf[abandoned] = scouts.buf
            self.trials[abandoned] = 0
