#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np

from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native.agent cimport LegacyNativeAgent
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget
from clypto.native.collection.vectorize.swarm_based.PSO.P_PSO cimport P_PSO


cdef class C_PSO(P_PSO):
    """
    The original version of: Chaos Particle Swarm Optimization (C-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1.0, 3.0] local coefficient, default = 2.05
        + c2 (float): [1.0, 3.0] global coefficient, default = 2.05
        + w_min (float): [0.1, 0.4], Weight min of bird, default = 0.4
        + w_max (float): [0.4, 2.0], Weight max of bird, default = 0.9

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.C_PSO(epoch=1000, pop_size=50, c1=2.05, c2=2.05, w_min=0.4, w_max=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Liu, B., Wang, L., Jin, Y.H., Tang, F. and Huang, D.X., 2005. Improved particle swarm optimization
    combined with chaos. Chaos, Solitons & Fractals, 25(5), pp.1261-1271.
    """

    cdef public double c1
    cdef public double c2
    cdef public double w_min
    cdef public double w_max
    cdef public int N_CLS
    cdef public object dyn_lb
    cdef public object dyn_ub

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        w_min: float = 0.4,
        w_max: float = 0.9,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient, default = 2.05
            c2: [0-2] global coefficient, default = 2.05
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "c1", "c2", "w_min", "w_max"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.w_min = cy.validator(float, w_min, (0, 0.5), "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")

    cdef void initialize_variables(self):
        self.v_max = 0.5 * (self.problem.ub - self.problem.lb)
        self.v_min = -self.v_max
        self.N_CLS = int(self.pop_size / 5)  # Number of chaotic local searches
        self.dyn_lb = self.problem.lb.copy()
        self.dyn_ub = self.problem.ub.copy()

    cdef object get_weights__(self, object fit, object fit_avg, object fit_min):
        temp1 = self.w_min + (self.w_max - self.w_min) * (fit - fit_min) / (fit_avg - fit_min)
        if self.problem.minmax == "min":
            return temp1 if fit <= fit_avg else self.w_max
        return self.w_max if fit <= fit_avg else temp1

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef NativePopulation merged
        cdef LegacyNativeAgent g_best
        cdef NativeTarget target_best
        cdef Py_ssize_t start, stop, n = pop.n
        list_fits = pop.F.tolist()
        fit_avg = np.mean(list_fits)
        fit_min = np.min(list_fits)
        w = np.array([self.get_weights__(fit, fit_avg, fit_min) for fit in list_fits])[:, None]
        R = self.generator.random((n, 2))  # per agent: two scalar draws
        r1 = (self.c1 * R[:, 0])[:, None]
        r2 = (self.c2 * R[:, 1])[:, None]
        X, V, P = pop.X, pop.field("V"), pop.field("P")
        for start, stop in self.chunks(n):
            g = np.array(self.g_best_x(), dtype=float)
            Xs = X[start:stop]
            v_new = (
                w[start:stop] * V[start:stop]
                + r1[start:stop] * (P[start:stop] - Xs)
                + r2[start:stop] * (g - Xs)
            )
            v_new = np.clip(v_new, self.v_min, self.v_max)
            x_new = Xs + v_new
            V[start:stop] = v_new
            cand.X[start:stop] = self.correct_solution(np.clip(x_new, self.dyn_lb, self.dyn_ub))
            self.evaluate(cand, start, stop)
            self.accept(cand, start, stop)

        ## Implement chaostic local search for the best solution
        g_best = self.current_g_best().copy()
        cx_best_0 = (g_best.solution - self.problem.lb) / (self.problem.ub - self.problem.lb)  # Eq. 7
        cx_best_1 = 4 * cx_best_0 * (1 - cx_best_0)  # Eq. 6
        x_best = self.problem.lb + cx_best_1 * (self.problem.ub - self.problem.lb)  # Eq. 8
        x_best = self.correct_solution(x_best)
        target_best = self.get_target(x_best)
        # The classic engine compared with the default minmax="min" here.
        if cy.compare_target(target_best, g_best.target, "min"):
            g_best.solution = x_best
            g_best.target = target_best

        r = self.generator.random()
        bound_min = np.stack([self.dyn_lb, g_best.solution - r * (self.dyn_ub - self.dyn_lb)])
        self.dyn_lb = np.max(bound_min, axis=0)
        bound_max = np.stack([self.dyn_ub, g_best.solution + r * (self.dyn_ub - self.dyn_lb)])
        self.dyn_ub = np.min(bound_max, axis=0)

        merged = pop.concat(self.generate_population(self.pop_size - self.N_CLS))
        self.pop = merged.take(self.sorted_order(merged)[:self.pop_size])
