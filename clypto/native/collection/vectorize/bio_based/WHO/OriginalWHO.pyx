#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalWHO(VectorizeOptimizer):
    """
    The original version of: Wildebeest Herd Optimization (WHO)

    Links:
        1. https://doi.org/10.3233/JIFS-190495

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_explore_step (int): [2, 10] -> better [2, 4], number of exploration step
        + n_exploit_step (int): [2, 10] -> better [2, 4], number of exploitation step
        + eta (float): (0, 1.0) -> better [0.05, 0.5], learning rate
        + p_hi (float): (0, 1.0) -> better [0.7, 0.95], the probability of wildebeest move to another position based on herd instinct
        + local_alpha (float): (0, 3.0) -> better [0.5, 0.9], control local movement (alpha 1)
        + local_beta (float): (0, 3.0) -> better [0.1, 0.5], control local movement (beta 1)
        + global_alpha (float): (0, 3.0) -> better [0.1, 0.5], control global movement (alpha 2)
        + global_beta (float): (0, 3.0), control global movement (beta 2)
        + delta_w (float): (0.5, 5.0) -> better [1.0, 2.0], dist to worst
        + delta_c (float): (0.5, 5.0) -> better [1.0, 2.0], dist to best

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.bio_based import WHO    >>> import numpy as np
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
    >>> model = WHO.OriginalWHO(epoch=1000, pop_size=50, n_explore_step = 3, n_exploit_step = 3, eta = 0.15, p_hi = 0.9,
    >>>                         local_alpha=0.9, local_beta=0.3, global_alpha=0.2, global_beta=0.8, delta_w=2.0, delta_c=2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Amali, D. and Dinakaran, M., 2019. Wildebeest herd optimization: a new global optimization algorithm inspired
    by wildebeest herding behaviour. Journal of Intelligent & Fuzzy Systems, 37(6), pp.8063-8076.
    """

    cdef public object n_explore_step
    cdef public object n_exploit_step
    cdef public object eta
    cdef public object p_hi
    cdef public object local_alpha
    cdef public object local_beta
    cdef public object global_alpha
    cdef public object global_beta
    cdef public object delta_w
    cdef public object delta_c

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        n_explore_step = 3,
        n_exploit_step = 3,
        eta = 0.15,
        p_hi = 0.9,
        local_alpha = 0.9,
        local_beta = 0.3,
        global_alpha = 0.2,
        global_beta = 0.8,
        delta_w = 2.0,
        delta_c = 2.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_explore_step (int): default = 3, number of exploration step
            n_exploit_step (int): default = 3, number of exploitation step
            eta (float): default = 0.15, learning rate
            p_hi (float): default = 0.9, the probability of wildebeest move to another position based on herd instinct
            local_alpha (float): control local movement (alpha 1)
            local_beta (float): control local movement (beta 1)
            global_alpha (float): control global movement (alpha 2)
            global_beta (float): control global movement (beta 2)
            delta_w (float): dist to worst
            delta_c (float): dist to best
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "n_explore_step",
                "n_exploit_step",
                "eta",
                "p_hi",
                "local_alpha",
                "local_beta",
                "global_alpha",
                "global_beta",
                "delta_w",
                "delta_c",
            ],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.n_explore_step = cy.validator(int, n_explore_step, [2, 10], "n_explore_step")
        self.n_exploit_step = cy.validator(int, n_exploit_step, [2, 10], "n_exploit_step")
        self.eta = cy.validator(float, eta, (0, 1.0), "eta")
        self.p_hi = cy.validator(float, p_hi, (0, 1.0), "p_hi")
        self.local_alpha = cy.validator(float, local_alpha, (0, 3.0), "local_alpha")
        self.local_beta = cy.validator(float, local_beta, (0, 3.0), "local_beta")
        self.global_alpha = cy.validator(float, global_alpha, (0, 3.0), "global_alpha")
        self.global_beta = cy.validator(float, global_beta, (0, 3.0), "global_beta")
        self.delta_w = cy.validator(float, delta_w, (0.5, 5.0), "delta_w")
        self.delta_c = cy.validator(float, delta_c, (0.5, 5.0), "delta_c")

    def _evolve(self, int epoch_c):
        # Agents are compared and replaced as they move (the later steps see the rows updated
        # before), so the loops run on the buffer rows; swarm modes collect the candidates.
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativePopulation local, child
        cdef NativeTarget tar
        cdef Py_ssize_t idx, jdx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        sense = self.problem.sense
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        Xp = pop.X
        ## Begin the Wildebeest Herd Optimization process
        for idx in range(0, self.pop_size):
            ### 1. Local movement (Milling behaviour)
            k = self.n_explore_step
            R = self.generator.random((k, 1 + d))  # per step: uniform(), then uniform(lb, ub)
            temp = Xp[idx] + self.eta * R[:, :1] * (lb + (ub - lb) * R[:, 1:])
            local = self.new_population(self._correct_solution(temp))
            best_local = local.X[self.sorted_order(local)[0]]
            temp = self.local_alpha * best_local + self.local_beta * (Xp[idx] - best_local)
            ops.commit(self, pop, cand, idx, self._correct_solution(temp), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
        for idx in range(0, self.pop_size):
            ### 2. Herd instinct
            idr = self.generator.choice(range(0, self.pop_size))
            if self._compare_fitness(pop.F[idr], pop.F[idx], sense) and self.generator.random() < self.p_hi:
                temp = self.global_alpha * Xp[idx] + self.global_beta * Xp[idr]
                pos_new = self._correct_solution(temp)
                tar = self._get_target(pos_new)
                if self._compare_fitness(tar.fitness, pop.F[idx], sense):
                    ops.set_row(pop, idx, pos_new, tar)

        order = self.sorted_order(pop)
        g_best, g_worst = Xp[order[0]].copy(), Xp[order[n - 1]].copy()
        pop_child = []  # swarm modes: the stored solutions of every candidate
        for idx in range(0, self.pop_size):
            dist_to_worst = np.linalg.norm(Xp[idx] - g_worst)
            dist_to_best = np.linalg.norm(Xp[idx] - g_best)
            ### 3. Starvation avoidance
            if dist_to_worst < self.delta_w:
                temp = Xp[idx] + self.generator.uniform() * (ub - lb) * self.generator.uniform(lb, ub)
                pos_new = self._correct_solution(temp)
                if swarm:
                    pop_child.append(pos_new)
                else:
                    tar = self._get_target(pos_new)
                    if self._compare_fitness(tar.fitness, pop.F[idx], sense):
                        ops.set_row(pop, idx, pos_new, tar)
            ### 4. Population pressure
            if 1.0 < dist_to_best and dist_to_best < self.delta_c:
                temp = g_best + self.eta * self.generator.uniform(lb, ub)
                pos_new = self._correct_solution(temp)
                if swarm:
                    pop_child.append(pos_new)
                else:
                    tar = self._get_target(pos_new)
                    if self._compare_fitness(tar.fitness, pop.F[idx], sense):
                        ops.set_row(pop, idx, pos_new, tar)
            ### 5. Herd social memory (the classic agent keeps the uncorrected position)
            for jdx in range(0, self.n_exploit_step):
                temp = g_best + 0.1 * self.generator.uniform(lb, ub)
                pos_new = self._correct_solution(temp)
                if swarm:
                    pop_child.append(temp)
                else:
                    tar = self._get_target(pos_new)
                    if self._compare_fitness(tar.fitness, pop.F[idx], sense):
                        ops.set_row(pop, idx, temp, tar)
        if swarm:
            child = self.new_population(np.array(pop_child))
            child = child.take(self.sorted_order(child)[:self.pop_size])
            ops.greedy(self, child)
