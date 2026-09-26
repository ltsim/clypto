#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native.chaotic import ChaoticMap as CM
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class ChaoticGWO(LegacyNativeOptimizer):
    """
    The original version of: Chaotic-based Grey Wolf Optimizer (Chaotic-GWO or C-GWO)

    Links:
        1. https://doi.org/10.1016/j.jcde.2017.02.005

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import GWO    >>> import numpy as np
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
    >>> model = GWO.ChaoticGWO(epoch=1000, pop_size=50, chaotic_name="chebyshev", initial_chaotic_value=0.7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kohli, M., & Arora, S. (2018). Chaotic grey wolf optimization algorithm for constrained optimization problems. Journal of computational design and engineering, 5(4), 458-472.
    """

    CHAOTIC_MAPS = {
        "bernoulli": CM.bernoulli_map,
        "logistic": CM.logistic_map,
        "chebyshev": CM.chebyshev_map,
        "circle": CM.circle_map,
        "cubic": CM.cubic_map,
        "icmic": CM.icmic_map,
        "piecewise": CM.piecewise_map,
        "singer": CM.singer_map,
        "sinusoidal": CM.sinusoidal_map,
        "tent": CM.tent_map,
    }

    cdef public object chaotic_name
    cdef public double initial_chaotic_value
    cdef public object chao_value
    cdef public object chao_func

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        chaotic_name: str = "chebyshev",
        initial_chaotic_value: float = 0.7,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            chaotic_name (str): name of chaotic map to use, default = "chebyshev"
            initial_chaotic_value (float): initial value for chaotic map, default = 0.7
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "chaotic_name", "initial_chaotic_value"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.chaotic_name = cy.validator(str, chaotic_name, ChaoticGWO.CHAOTIC_MAPS.keys(), "chaotic_name")
        self.initial_chaotic_value = cy.validator(float, initial_chaotic_value, [0.0, 1.0], "initial_chaotic_value")

    cdef void initialize_variables(self):
        self.chao_value = self.initial_chaotic_value
        self.chao_func = ChaoticGWO.CHAOTIC_MAPS[self.chaotic_name]

    def _update_chao_value(self):
        chao_value = self.chao_func(self.chao_value)
        # Ensure chaotic value stays in [0, 1]
        self.chao_value = np.clip(chao_value, 0, 1)

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        # linearly decreased from 2 to 0
        a = 2 - 2.0 * epoch / self.epoch
        best = pop.X[self.sorted_order(pop)[:3]][None]
        cv = np.empty(n)
        for idx in range(n):  # the chaotic value advances once per agent
            self._update_chao_value()
            cv[idx] = self.chao_value
        cv = cv[:, None, None]
        R = self.generator.random((n, 6, d))  # per agent: A1..A3 then C1..C3 draws
        A = a * (2 * R[:, :3] * cv - 1)
        C = 2 * R[:, 3:] * cv
        Xs = best - A * np.abs(C * best - pop.X[:, None, :])
        cand.X[:] = self.correct_solution((Xs[:, 0] + Xs[:, 1] + Xs[:, 2]) / 3.0)
        self.evaluate(cand, 0, n)
        ops.accept(self, cand)
