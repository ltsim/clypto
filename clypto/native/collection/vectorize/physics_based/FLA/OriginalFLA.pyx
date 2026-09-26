#!/usr/bin/env python
# Created by "Thieu" at 21:00, 14/03/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class OriginalFLA(VectorizeOptimizer):
    """
    The original version of: Fick's Law Algorithm (FLA)

    Notes:
        1. The algorithm contains a high number of parameters, some of which may be unnecessary.
        2. Despite the complexity of the algorithms, they may not perform optimally and could potentially become trapped in local optima.
        3. Division by the fitness value may cause overflow issues to arise.
        4. https://www.mathworks.com/matlabcentral/fileexchange/121033-fick-s-law-algorithm-fla

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + C1 (float): factor C1, default=0.5
        + C2 (float): factor C2, default=2.0
        + C3 (float): factor C3, default=0.1
        + C4 (float): factor C4, default=0.2
        + C5 (float): factor C5, default=2.0
        + DD (float): factor D in the paper, default=0.01

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import FLA    >>> import numpy as np
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
    >>> model = FLA.OriginalFLA(epoch=1000, pop_size=50, C1 = 0.5, C2 = 2.0, C3 = 0.1, C4 = 0.2, C5 = 2.0, DD = 0.01)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F. A., Mostafa, R. R., Hussien, A. G., Mirjalili, S., & Sallam, K. M. (2023). Fick’s Law Algorithm: A physical
    law-based algorithm for numerical optimization. Knowledge-Based Systems, 260, 110146.
    """

    cdef public double C1
    cdef public double C2
    cdef public double C3
    cdef public double C4
    cdef public double C5
    cdef public double DD
    cdef public object xss
    cdef public object n1
    cdef public object n2
    cdef public object pop1
    cdef public object pop2
    cdef public object best1
    cdef public object best2
    cdef public object fsss

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        C1: float = 0.5,
        C2: float = 2.0,
        C3: float = 0.1,
        C4: float = 0.2,
        C5: float = 2.0,
        DD: float = 0.01,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            C1 (float): factor C1, default=0.5
            C2 (float): factor C2, default=2.0
            C3 (float): factor C3, default=0.1
            C4 (float): factor C4, default=0.2
            C5 (float): factor C5, default=2.0
            DD (float): factor D in the paper, default=0.01
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "C1", "C2", "C3", "C4", "C5", "DD"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [10, 10000], "pop_size")
        self.C1 = cy.validator(float, C1, (-100.0, 100.0), "C1")
        self.C2 = cy.validator(float, C2, (-100.0, 100.0), "C2")
        self.C3 = cy.validator(float, C3, (-100.0, 100.0), "C3")
        self.C4 = cy.validator(float, C4, (-100.0, 100.0), "C4")
        self.C5 = cy.validator(float, C5, (-100.0, 100.0), "C5")
        self.DD = cy.validator(float, DD, (-100.0, 100.0), "DD")

    def _before_main_loop(self):
        self.xss = self.pop.take(self.sorted_order(self.pop))
        self.n1 = int(np.round(self.pop_size / 2))
        self.n2 = self.pop_size - self.n1
        self.split_teams__()

    def split_teams__(self):
        """Two teams (first n1 / last n2 rows), their best agents and the reference fitness."""
        self.pop1 = self.pop.take(np.arange(self.n1))
        self.pop2 = self.pop.take(np.arange(self.n1, self.pop.n))
        self.best1 = self.pop1.agent(self.sorted_order(self.pop1)[0])
        self.best2 = self.pop2.agent(self.sorted_order(self.pop2)[0])
        if self._compare_fitness(self.best1.target.fitness, self.best2.target.fitness, self.problem.sense):
            self.fsss = self.best1.target.fitness
        else:
            self.fsss = self.best2.target.fitness

    def _evolve(self, int epoch):
        # Candidates are built team by team (draws in the classic order); evaluation is batched
        # and each agent is replaced when its candidate is better (compare_target).
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativePopulation P1 = self.pop1
        cdef NativePopulation P2 = self.pop2
        cdef Py_ssize_t idx
        g_best = np.array(self.g_best_x())
        pos_list = np.array(pop.X)
        pos1_list = np.array(P1.X)
        pos2_list = np.array(P2.X)
        xm1 = np.mean(pos1_list, axis=0)
        xm2 = np.mean(pos2_list, axis=0)
        xm = np.mean(pos_list, axis=0)
        tf = np.sinh(epoch / self.epoch) ** self.C1
        pop_new = []
        if tf < 0.9:
            dof = np.exp(-(self.C2 * tf - self.generator.random())) ** self.C2
            tdo = self.C5 * tf - self.generator.random()
            if tdo < self.generator.random():
                m1n, m2n = self.C3 * self.n1, self.C4 * self.n1
                nt12 = int(np.round((m2n - m1n) * self.generator.random() + m1n))
                for idx in range(0, nt12):
                    dfg = self.generator.integers(1, 3)
                    jj = (
                            -self.DD
                            * (xm2 - xm1)
                            / (
                                    np.linalg.norm(
                                        self.best2.solution - P1.X[idx]
                                    )
                                    + self.EPSILON
                            )
                    )
                    pos_new = self.best2.solution + dfg * dof * self.generator.random(
                        self.problem.n_dims
                    ) * (jj * self.best2.solution - P1.X[idx])
                    pop_new.append(self._correct_solution(pos_new))
                for idx in range(nt12, self.n1):
                    tt = P1.X[idx] + dof * (
                            self.generator.random(self.problem.n_dims)
                            * (self.problem.bounds.up - self.problem.bounds.low)
                            + self.problem.bounds.low
                    )
                    pp = self.generator.random(self.problem.n_dims)
                    pos_new = np.where(
                        pp < 0.8,
                        self.best1.solution,
                        np.where(pp >= 0.9, P1.X[idx], tt),
                    )
                    pop_new.append(self._correct_solution(pos_new))
                for idx in range(0, self.n2):
                    pos_new = self.best2.solution + dof * (
                            self.generator.random(self.problem.n_dims)
                            * (self.problem.bounds.up - self.problem.bounds.low)
                            + self.problem.bounds.low
                    )
                    pop_new.append(self._correct_solution(pos_new))
            else:
                m1n, m2n = 0.1 * self.n2, 0.2 * self.n2
                nt12 = int(np.round((m2n - m1n) * self.generator.random() + m1n))
                for idx in range(0, nt12):
                    dfg = self.generator.integers(1, 3)
                    jj = (
                            -self.DD
                            * (xm1 - xm2)
                            / (
                                    np.linalg.norm(
                                        self.best1.solution - P2.X[idx]
                                    )
                                    + self.EPSILON
                            )
                    )
                    pos_new = self.best1.solution + dfg * dof * self.generator.random(
                        self.problem.n_dims
                    ) * (jj * self.best1.solution - P2.X[idx])
                    pop_new.append(self._correct_solution(pos_new))
                for idx in range(nt12, self.n2):
                    tt = P2.X[idx] + dof * (
                            self.generator.random(self.problem.n_dims)
                            * (self.problem.bounds.up - self.problem.bounds.low)
                            + self.problem.bounds.low
                    )
                    pp = self.generator.random(self.problem.n_dims)
                    pos_new = np.where(
                        pp < 0.8,
                        self.best2.solution,
                        np.where(pp >= 0.9, P2.X[idx], tt),
                    )
                    pop_new.append(self._correct_solution(pos_new))
                for idx in range(0, self.n1):
                    pos_new = self.best1.solution + dof * (
                            self.generator.random(self.problem.n_dims)
                            * (self.problem.bounds.up - self.problem.bounds.low)
                            + self.problem.bounds.low
                    )
                    pop_new.append(self._correct_solution(pos_new))
        else:  # Equilibrium operator (EO)
            if tf <= 1:
                for idx in range(0, self.n1):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(self.best1.solution - P1.X[idx])
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (self.best1.solution - xm1) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(
                        -self.best1.target.fitness
                        / (P1.F[idx] + self.EPSILON)
                    )
                    qeo = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = (
                            self.best1.solution
                            + qeo * P1.X[idx]
                            + qeo * (ms * self.best1.solution - P1.X[idx])
                    )
                    pop_new.append(self._correct_solution(pos_new))
                for idx in range(0, self.n2):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(self.best2.solution - P2.X[idx])
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (self.best2.solution - xm2) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(
                        -self.best2.target.fitness
                        / (P2.F[idx] + self.EPSILON)
                    )
                    qeo = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = (
                            self.best2.solution
                            + qeo * P2.X[idx]
                            + qeo * (ms * self.best2.solution - P2.X[idx])
                    )
                    pop_new.append(self._correct_solution(pos_new))
            else:  # Steady state operator (SSO)
                for idx in range(0, self.n1):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(
                        g_best - P1.X[idx]
                    )
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (xm - xm1) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(
                        -self.fsss / (P1.F[idx] + self.EPSILON)
                    )
                    qg = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = (
                            g_best
                            + qg * P1.X[idx]
                            + qg * (ms * self.best1.solution - P1.X[idx])
                    )
                    pop_new.append(self._correct_solution(pos_new))
                for idx in range(0, self.n2):
                    dfg = self.generator.integers(1, 3)
                    tttt = np.linalg.norm(
                        g_best - P2.X[idx]
                    )
                    if tttt == 0:
                        jj = 0
                    else:
                        jj = -self.DD * (xm - xm2) / tttt
                    drf = np.exp(-jj / tf)
                    ms = np.exp(
                        -self.fsss / (P2.F[idx] + self.EPSILON)
                    )
                    qg = dfg * drf * self.generator.random(self.problem.n_dims)
                    pos_new = (
                            g_best
                            + qg * P2.X[idx]
                            + qg * (ms * g_best - P2.X[idx])
                    )
                    pop_new.append(self._correct_solution(pos_new))
        cand.X[:] = np.array(pop_new)
        self.evaluate(cand, 0, pop.n)
        better = cand.F < pop.F
        if self.problem.sense != "min":
            better = ~better
        rows = np.flatnonzero(better)
        pop.buf[rows] = cand.buf[rows]
        self.split_teams__()
