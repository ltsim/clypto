#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class OriginalBeesA(LegacyNativeOptimizer):
    """
    The original version of: Bees Algorithm (BeesA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/B978008045157250081X
        2. https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + selected_site_ratio (float): default = 0.5
        + elite_site_ratio (float): default = 0.4
        + selected_site_bee_ratio (float): default = 0.1
        + elite_site_bee_ratio (float): default = 2.0
        + dance_radius (float): default = 0.1
        + dance_reduction (float): default = 0.99

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BeesA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BeesA.OriginalBeesA(epoch=1000, pop_size=50, selected_site_ratio=0.5, elite_site_ratio=0.4,
    >>>         selected_site_bee_ratio=0.1, elite_site_bee_ratio=2.0, dance_radius=0.1, dance_reduction=0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T., Ghanbarzadeh, A., Koç, E., Otri, S., Rahim, S. and Zaidi, M., 2006. The bees algorithm—a novel tool
    for complex optimisation problems. In Intelligent production machines and systems (pp. 454-459). Elsevier Science Ltd.
    """

    cdef public object selected_site_ratio
    cdef public object elite_site_ratio
    cdef public object selected_site_bee_ratio
    cdef public object elite_site_bee_ratio
    cdef public object dance_radius
    cdef public object dance_reduction
    cdef public object dyn_radius
    cdef public object n_selected_bees
    cdef public object n_elite_bees
    cdef public object n_selected_bees_local
    cdef public object n_elite_bees_local

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        selected_site_ratio: float = 0.5,
        elite_site_ratio: float = 0.4,
        selected_site_bee_ratio: float = 0.1,
        elite_site_bee_ratio: float = 2.0,
        dance_radius: float = 0.1,
        dance_reduction: float = 0.99,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            selected_site_ratio (float):
            elite_site_ratio (float):
            selected_site_bee_ratio (float):
            elite_site_bee_ratio (float):
            dance_radius (float):
            dance_reduction (float):
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "selected_site_ratio",
                "elite_site_ratio",
                "selected_site_bee_ratio",
                "elite_site_bee_ratio",
                "dance_radius",
                "dance_reduction",
            ],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.selected_site_ratio = cy.validator(float, selected_site_ratio, (0, 1.0), "selected_site_ratio")
        self.elite_site_ratio = cy.validator(float, elite_site_ratio, (0, 1.0), "elite_site_ratio")
        self.selected_site_bee_ratio = cy.validator(float, selected_site_bee_ratio, (0, 1.0), "selected_site_bee_ratio")
        self.elite_site_bee_ratio = cy.validator(float, elite_site_bee_ratio, (0, 3.0), "elite_site_bee_ratio")
        self.dance_radius = cy.validator(float, dance_radius, (0, 1.0), "dance_radius")
        self.dance_reduction = cy.validator(float, dance_reduction, (0, 1.0), "dance_reduction")
        self.dyn_radius = self.dance_radius
        self.n_selected_bees = int(round(self.selected_site_ratio * self.pop_size))
        self.n_elite_bees = int(round(self.elite_site_ratio * self.n_selected_bees))
        self.n_selected_bees_local = int(
                    round(self.selected_site_bee_ratio * self.pop_size)
                )
        self.n_elite_bees_local = int(
                    round(self.elite_site_bee_ratio * self.n_selected_bees_local)
                )

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, fresh
        cdef Py_ssize_t n = pop.n, d = pop.d, ne = self.n_elite_bees, ns = self.n_selected_bees
        cdef object rng = self.generator
        # elite sites and selected sites recruit bees that dance around them (one random coordinate moves)
        counts = np.zeros(n, dtype=int)
        counts[:ne] = self.n_elite_bees_local
        counts[ne:ns] = self.n_selected_bees_local
        parent = np.repeat(np.arange(n), counts)
        pos = np.array(pop.X[parent])
        pos[np.arange(len(parent)), rng.integers(0, d, size=len(parent))] += self.dyn_radius * rng.uniform(-1, 1, len(parent))
        cand = pop.take(parent)
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, len(parent))
        ops.scatter(self, cand, parent)  # each site keeps its best neighbour if it improves it
        # the remaining bees scout new random sources
        if ns < n:
            fresh = pop.take(np.arange(ns, n))
            fresh.X[:] = self.problem.lb + rng.random((n - ns, d)) * (self.problem.ub - self.problem.lb)
            self.evaluate(fresh, 0, n - ns)
            pop.buf[ns:] = fresh.buf
        self.dyn_radius = self.dance_reduction * self.dance_radius
