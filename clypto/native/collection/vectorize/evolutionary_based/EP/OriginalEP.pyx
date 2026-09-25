#!/usr/bin/env python
# Created by "Thieu" at 19:27, 10/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np



from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalEP(LegacyNativeOptimizer):
    """
    The original version of: Evolutionary Programming (EP)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/evolution/evolutionary_programming.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + bout_size (float): [0.05, 0.2], percentage of child agents implement tournament selection

    Examples
    ~~~~~~~~
    >>> from clypto.collection.evolutionary_based import EP    >>> import numpy as np
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
    >>> model = EP.OriginalEP(epoch=1000, pop_size=50, bout_size = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Yao, X., Liu, Y. and Lin, G., 1999. Evolutionary programming made faster.
    IEEE Transactions on Evolutionary computation, 3(2), pp.82-102.
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        bout_size: float = 0.05,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size (miu in the paper), default = 100
            bout_size (float): percentage of child agents implement tournament selection
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "bout_size"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.bout_size = cy.validator(float, bout_size, (0, 1.0), "bout_size")

    cdef list layout(self, Py_ssize_t d, Py_ssize_t m):
        return [("S", d), ("WIN", 1)]  # mutation strategy and tournament wins

    cdef void initialize_variables(self):
        self.n_bout_size = int(self.bout_size * self.pop_size)
        self.distance = 0.05 * (self.problem.ub - self.problem.lb)

    cdef void init_fields(self, NativePopulation pop):
        pop.field("S")[:] = self.generator.uniform(0, self.distance, (pop.n, pop.d))
        pop.field("WIN")[:] = 0

    def tournament__(self, NativePopulation pop):
        """Every agent enters n_bout_size tournaments against random agents; the winner scores."""
        cdef Py_ssize_t i, idx, rand_idx, m = pop.n
        F, win = pop.F, pop.field("WIN")[:, 0]
        minmax = self.problem.minmax
        for i in range(0, m):
            ## Tournament winner (Tried with bout_size times)
            for idx in range(0, self.n_bout_size):
                rand_idx = self.generator.integers(0, m)
                if self.compare_fitness(F[i], F[rand_idx], minmax):
                    win[i] += 1
                else:
                    win[rand_idx] += 1

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation child = pop.empty_like()
        cdef NativePopulation both
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        Xp, S = pop.X, pop.field("S")
        # offspring: the strategy draw of generate_empty_agent is made and dropped, as in the classic code
        for idx in range(0, self.pop_size):
            pos_new = Xp[idx] + S[idx] * self.generator.normal(0, 1.0, d)
            child.X[idx] = self.correct_solution(pos_new)
            self.generator.uniform(0, self.distance, d)
            child.field("S")[idx] = S[idx] + self.generator.normal(0, 1.0, d) * np.abs(S[idx]) ** 0.5
        child.field("WIN")[:] = 0
        self.evaluate(child, 0, n)
        # Update the global best
        both = child.take(self.sorted_order(child)).concat(pop)
        self.tournament__(both)
        order = sorted(range(both.n), key=lambda i: both.field("WIN")[i, 0], reverse=True)
        self.pop = both.take(order[:self.pop_size])
