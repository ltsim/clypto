#!/usr/bin/env python
# Created by "Thieu" at 06:25, 17/06/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalTS(LegacyNativeOptimizer):
    """
    The original version of: Tabu Search (TS)

    Notes:
        + The pop_size is not an official parameter in this algorithm. However, we need it here to adapt to Mealpy library.
        + You should set pop_size = 2 to reduce the initial computation for the initial population of this algorithm.
        + The perturbation_scale is important parameter that effect the most to this algorithm.

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + tabu_size (int): [5, 10] Maximum size of the tabu list.
        + neighbour_size (int): [5, 100] Size of the neighborhood for generating candidate solutions, Default: 10
        + perturbation_scale (float): [0.01 - 1] Scale of the perturbations for generating candidate solutions. default = 0.05

    Examples
    ~~~~~~~~
    >>> from clypto.collection.math_based import TS    >>> import numpy as np
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
    >>> model = TS.OriginalTS(epoch=1000, pop_size=50, tabu_size = 5, neighbour_size = 20, perturbation_scale = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hajji, O., Brisset, S., & Brochet, P. (2004). A new tabu search method for optimization
    with continuous parameters. IEEE Transactions on Magnetics, 40(2), 1184-1187.
    """

    cdef public int tabu_size
    cdef public int neighbour_size
    cdef public double perturbation_scale
    cdef public object x
    cdef public object tabu_list

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 2,
        tabu_size: int = 5,
        neighbour_size: int = 10,
        perturbation_scale: float = 0.05,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size. **This is not an official parameter**. So default = 2.
            tabu_size (int): Maximum size of the tabu list.
            neighbour_size (int): Size of the neighborhood for generating candidate solutions, Default: 10
            perturbation_scale (float): Scale of the perturbations for generating candidate solutions. default = 0.05
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "tabu_size", "neighbour_size", "perturbation_scale"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [2, 10000], "pop_size")
        self.tabu_size = cy.validator(int, tabu_size, [2, 10000], "tabu_size")
        self.neighbour_size = cy.validator(int, neighbour_size, [2, 10000], "neighbour_size")
        self.perturbation_scale = cy.validator(float, perturbation_scale, (0, 100), "perturbation_scale")

    cdef void before_main_loop(self):
        self.x = np.array(self.g_best.solution)
        self.tabu_list = []
        self.pop = self.pop.take(np.array([], dtype=int))  # the search keeps its best moves instead

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        # Generate candidate solutions by perturbing the current solution
        candidates = self.generator.normal(
            loc=self.x,
            scale=self.perturbation_scale,
            size=(self.neighbour_size, self.problem.n_dims),
        )
        # Evaluate candidate solutions and select best move
        list_candidates = []
        for candidate in candidates:
            pos_new = self.correct_solution(candidate)
            if np.allclose(pos_new, self.x):
                continue
            if tuple(pos_new) in self.tabu_list:
                continue
            list_candidates.append(pos_new)
        cand = self.new_population(np.array(list_candidates))
        best = self.sorted_order(cand)[0]
        self.x = cand.X[best].copy()
        # Update tabu list
        self.tabu_list.append(tuple(self.x))
        pop = pop.concat(cand.take([best]))
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)
            pop = pop.take(np.arange(1, pop.n))
        self.pop = pop
