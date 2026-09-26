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


cdef class OriginalMRFO(LegacyNativeOptimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import MRFO    >>> import numpy as np
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
    >>> model = MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    cdef public object somersault_range

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        somersault_range: float = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "somersault_range"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.somersault_range = cy.validator(float, somersault_range, [1.0, 5.0], "somersault_range")

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        lb, ub = self.problem.lb, self.problem.ub
        prev = np.vstack([X[:1], X[:-1]])  # the previous agent (the first one uses the leader instead)
        first = (np.arange(n) == 0)[:, None]
        # chain foraging / cyclone foraging
        r1 = rng.uniform(size=(n, 1))
        beta = 2 * np.exp(r1 * (self.epoch - epoch_c) / self.epoch) * np.sin(2 * np.pi * r1)
        x_rand = rng.uniform(lb, ub, (n, d))
        exploring = ((epoch_c + 1) / self.epoch < rng.random((n, 1)))
        lead = np.where(exploring, x_rand, g)  # the random point or the best
        cyclone = lead + rng.uniform(size=(n, 1)) * (np.where(first, lead, prev) - X) + beta * (lead - X)
        r = rng.uniform(size=(n, 1))
        alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
        chain = X + r * (np.where(first, g, prev) - X) + alpha * (g - X)
        ops.step(self, np.where(rng.random((n, 1)) < 0.5, cyclone, chain))
        # somersault foraging around the best agent found so far
        X = pop.X
        g = np.array(X[ops.best_row(self, self.pop)])
        ops.step(self, X + self.somersault_range * (rng.uniform(size=(n, 1)) * g - rng.uniform(size=(n, 1)) * X))
