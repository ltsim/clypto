#!/usr/bin/env python
# Created by "Thieu" at 12:09, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalCA(VectorizeOptimizer):
    """
    The original version of: Culture Algorithm (CA)

    Links:
        1. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + accepted_rate (float): [0.1, 0.5], probability of accepted rate, default: 0.15

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import CA    >>> import numpy as np
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
    >>> model = CA.OriginalCA(epoch=1000, pop_size=50, accepted_rate = 0.15)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, B., Zhao, L. and Lu, J.H., 2009, April. Wind power forecast using RBF network and culture algorithm.
    In 2009 International Conference on Sustainable Power Generation and Supply (pp. 1-6). IEEE.
    """

    cdef public object accepted_rate
    cdef public object dyn_belief_space
    cdef public object dyn_accepted_num

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        accepted_rate: float = 0.15,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            accepted_rate (float): probability of accepted rate, default: 0.15
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "accepted_rate"],
            sort_flag=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.accepted_rate = cy.validator(float, accepted_rate, (0, 1.0), "accepted_rate")

    def _initialize_variables(self):
        self.dyn_belief_space = {
            "lb": self.problem.bounds.low,
            "ub": self.problem.bounds.up,
        }
        self.dyn_accepted_num = int(self.accepted_rate * self.pop_size)

    def _evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation child, full
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        # children sampled inside the belief space
        child = pop.empty_like()
        child.X[:] = self._correct_solution(rng.uniform(self.dyn_belief_space["lb"], self.dyn_belief_space["ub"], (n, d)))
        self.evaluate(child, 0, n)
        full = pop.concat(child)
        # binary tournaments among parents and children
        id1 = rng.integers(0, 2 * n, size=n)
        id2 = (id1 + rng.integers(1, 2 * n, size=n)) % (2 * n)
        F = np.asarray(full.F)
        winner = np.where(ops.better(self, F[id1], F[id2]), id1, id2)
        self.pop = full.take(winner)
        self.pop = self.pop.take(self.sorted_order(self.pop))
        accepted = np.asarray(self.pop.X[:self.dyn_accepted_num])
        self.dyn_belief_space["lb"] = np.min(accepted, axis=0)
        self.dyn_belief_space["ub"] = np.max(accepted, axis=0)
