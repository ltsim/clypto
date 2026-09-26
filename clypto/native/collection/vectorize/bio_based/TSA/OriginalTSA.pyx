#!/usr/bin/env python
# Created by "Thieu" at 17:23, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalTSA(VectorizeOptimizer):
    """
    The original version: Tunicate Swarm Algorithm (TSA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0952197620300385?via%3Dihub
        2. https://www.mathworks.com/matlabcentral/fileexchange/75182-tunicate-swarm-algorithm-tsa

    Notes:
        1. This algorithm has some limitations
        2. The paper has several wrong equations in algorithm
        3. The implementation in Matlab code has some difference to the paper
        4. This algorithm shares some similarities with the Barnacles Mating Optimizer (BMO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import TSA    >>> import numpy as np
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
    >>> model = TSA.OriginalTSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm: A new bio-inspired
    based metaheuristic paradigm for global optimization. Engineering Applications of Artificial Intelligence, 90, 103541.
    """

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t n = pop.n, d = pop.d
        g_best = np.array(self.g_best_x())
        pmin, pmax = 1, 4
        # per agent: c3, c2, c1 (d draws each), then one draw for M
        R = self.generator.random((n, 3 * d + 1))
        c3, c2, c1 = R[:, :d], R[:, d:2 * d], R[:, 2 * d:3 * d]
        M = np.fix(pmin + R[:, 3 * d] * (pmax - pmin))[:, None]
        X = pop.X
        A = (c2 + c3 - 2 * c1) / M
        t1 = g_best + A * np.abs(g_best - c2 * X)
        t2 = g_best - A * np.abs(g_best - c2 * X)
        pos_new = np.where(c3 >= 0.5, t1, t2)
        pos_new[1:] = (pos_new[1:] + X[:-1]) / 2
        cand.X[:] = self._correct_solution(pos_new)
        self.evaluate(cand, 0, n)
        self.pop = cand
