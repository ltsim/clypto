#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalHGSO(LegacyNativeOptimizer):
    """
    The original version of: Henry Gas Solubility Optimization (HGSO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0167739X19306557

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_clusters (int): [2, 10], number of clusters, default = 2

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import HGSO    >>> import numpy as np
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
    >>> model = HGSO.OriginalHGSO(epoch=1000, pop_size=50, n_clusters = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. Henry gas solubility
    optimization: A novel physics-based algorithm. Future Generation Computer Systems, 101, pp.646-667.
    """

    cdef public int n_clusters
    cdef public object n_elements
    cdef public object T0
    cdef public object K
    cdef public object beta
    cdef public object alpha
    cdef public object epsilon
    cdef public object l1
    cdef public object l2
    cdef public object l3
    cdef public object H_j
    cdef public object P_ij
    cdef public object C_j
    cdef public object p_best
    cdef public object pop_group

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_clusters: int = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_clusters (int): number of clusters, default = 2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_clusters"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.n_clusters = cy.validator(int, n_clusters, [2, int(self.pop_size / 5)], "n_clusters")
        self.n_elements = int(self.pop_size / self.n_clusters)
        self.T0 = 298.15
        self.K = 1.0
        self.beta = 1.0
        self.alpha = 1
        self.epsilon = 0.05
        self.l1 = 5e-2
        self.l2 = 100.0
        self.l3 = 1e-2

    cdef void initialize_variables(self):
        self.H_j = self.l1 * self.generator.uniform()
        self.P_ij = self.l2 * self.generator.uniform()
        self.C_j = self.l3 * self.generator.uniform()
        self.pop_group, self.p_best = None, None

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        self.regroup__()

    def regroup__(self):
        """Split the population in n_clusters groups of n_elements agents and find each group's best."""
        cdef NativePopulation pop = self.pop
        cdef Py_ssize_t idx, m = self.n_elements
        self.pop_group = [pop.take(np.arange(idx * m, min((idx + 1) * m, pop.n))) for idx in range(self.n_clusters)]
        self.p_best = [g.agent(self.sorted_order(g)[0]) for g in self.pop_group]  # multiple element

    def flatten_group__(self):
        merged = self.pop_group[0]
        for idx in range(1, self.n_clusters):
            merged = merged.concat(self.pop_group[idx])
        return merged

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation grp, new, pop, out
        cdef Py_ssize_t idx, jdx
        g_best = np.array(self.g_best_x())
        ## Loop based on the number of cluster in swarm (number of gases type)
        for idx in range(self.n_clusters):
            grp = self.pop_group[idx]
            new = grp.empty_like()
            p_best = self.p_best[idx]
            ### Loop based on the number of individual in each gases type
            for jdx in range(self.n_elements):
                F = -1.0 if self.generator.uniform() < 0.5 else 1.0
                ##### Based on Eq. 8, 9, 10
                self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
                S_ij = self.K * self.H_j * self.P_ij
                gama = self.beta * np.exp(-((p_best.target.fitness + self.epsilon) / (grp.F[jdx] + self.epsilon)))
                pos_new = (
                        grp.X[jdx]
                        + F * self.generator.uniform() * gama * (p_best.solution - grp.X[jdx])
                        + F * self.generator.uniform() * self.alpha * (S_ij * g_best - grp.X[jdx])
                )
                new.X[jdx] = self.correct_solution(pos_new)
            self.evaluate(new, 0, new.n)
            self.pop_group[idx] = new
        pop = self.flatten_group__()
        self.pop = pop

        ## Update Henry's coefficient using Eq.8
        self.H_j = self.H_j * np.exp(-self.C_j * (1.0 / np.exp(-epoch / self.epoch) - 1.0 / self.T0))
        ## Update the solubility of each gas using Eq.9
        S_ij = self.K * self.H_j * self.P_ij
        ## Rank and select the number of worst agents using Eq. 11
        N_w = int(self.pop_size * (self.generator.uniform(0, 0.1) + 0.1))
        ## Update the position of the worst agents using Eq. 12
        sorted_id_pos = np.argsort(np.ascontiguousarray(pop.F))
        if N_w > 0:
            pos_new = self.correct_solution(self.generator.uniform(self.problem.lb, self.problem.ub, (N_w, pop.d)))
            out = self.new_population(pos_new)
            pop.buf[sorted_id_pos[:N_w]] = out.buf
        self.regroup__()
