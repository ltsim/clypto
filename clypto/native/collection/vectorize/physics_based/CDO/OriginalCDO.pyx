#!/usr/bin/env python
# Created by "Thieu" at 21:45, 13/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalCDO(LegacyNativeOptimizer):
    """
    The original version of: Chernobyl Disaster Optimizer (CDO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-023-08261-1
        2. https://www.mathworks.com/matlabcentral/fileexchange/124351-chernobyl-disaster-optimizer-cdo

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import CDO    >>> import numpy as np
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
    >>> model = CDO.OriginalCDO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Shehadeh, H. A. (2023). Chernobyl disaster optimizer (CDO): a novel meta-heuristic method
    for global optimization. Neural Computing and Applications, 1-17.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        b = pop.X[self.sorted_order(pop)[:3]]  # b1, b2, b3
        a = 3.0 - 3.0 * epoch / self.epoch
        a1 = np.log10((16000 - 1) * self.generator.random() + 16000)
        a2 = np.log10((270000 - 1) * self.generator.random() + 270000)
        a3 = np.log10((300000 - 1) * self.generator.random() + 300000)
        # per agent: r1, r2, the draw inside pa, then the same for pb and pc
        R = self.generator.random((n, 9, d))
        X = pop.X
        r1, r2, ra, r3, r4, rb, r5, r6, rc = (R[:, k] for k in range(9))
        pa = np.pi * r1 * r1 / (0.25 * a1) - a * ra
        c1 = r2 * r2 * np.pi
        alpha = np.abs(c1 * b[0] - X)
        pos_a = 0.25 * (b[0] - pa * alpha)

        pb = np.pi * r3 * r3 / (0.5 * a2) - a * rb
        c2 = r4 * r4 * np.pi
        beta = np.abs(c2 * b[1] - X)
        pos_b = 0.5 * (b[1] - pb * beta)

        pc = np.pi * r5 * r5 / a3 - a * rc
        c3 = r6 * r6 * np.pi
        gama = np.abs(c3 * b[2] - X)
        pos_c = b[2] - pc * gama

        cand.X[:] = self.correct_solution((pos_a + pos_b + pos_c) / 3)
        self.evaluate(cand, 0, n)
        self.pop = cand
