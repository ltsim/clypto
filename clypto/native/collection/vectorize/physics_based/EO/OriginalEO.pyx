#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalEO(LegacyNativeOptimizer):
    """
    The original version of: Equilibrium Optimizer (EO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2019.105190
        2. https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EO    >>> import numpy as np
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
    >>> model = EO.OriginalEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Faramarzi, A., Heidarinejad, M., Stephens, B. and Mirjalili, S., 2020. Equilibrium optimizer: A novel
    optimization algorithm. Knowledge-Based Systems, 191, p.105190.
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
        self.V = 1
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

    def make_equilibrium_pool__(self, NativePopulation best4):
        """The four best agents plus their mean position (evaluated), as one population."""
        pos_mean = np.mean(np.ascontiguousarray(best4.X), axis=0)
        pos_mean = self.correct_solution(pos_mean)
        return best4.concat(self.new_population(pos_mean[None]))

    def candidates__(self, NativePopulation pop, NativePopulation c_pool, int epoch):
        """EO moves of every agent."""
        rng = self.generator
        n, d = pop.n, pop.d
        t = (<object>(1 - epoch / self.epoch)) ** (<object>(self.a2 * epoch / self.epoch))
        lamda = rng.uniform(0, 1, (n, d))  # lambda in Eq. 11
        r = rng.uniform(0, 1, (n, d))  # r in Eq. 11
        c_eq = c_pool.X[rng.integers(0, c_pool.n, size=n)]  # random candidate from the pool
        r1 = rng.uniform(size=(n, 1))
        r2 = rng.uniform(size=(n, 1))  # r1, r2 in Eq. 15
        X = pop.X
        f = self.a1 * np.sign(r - 0.5) * (np.exp(-lamda * t) - 1.0)  # Eq. 11
        gcp = 0.5 * r1 * (r2 >= self.GP)  # Eq. 15
        g = gcp * (c_eq - lamda * X) * f  # Eqs. 13, 14
        return c_eq + (X - c_eq) * f + (g * self.V / lamda) * (1.0 - f)  # Eq. 16

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        c_pool = self.make_equilibrium_pool__(pop.take(self.sorted_order(pop)[:4]))
        ops.step(self, self.candidates__(pop, c_pool, epoch_c))
