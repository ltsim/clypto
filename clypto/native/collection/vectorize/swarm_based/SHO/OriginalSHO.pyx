#!/usr/bin/env python
# Created by "Thieu" at 10:55, 02/12/2019 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalSHO(LegacyNativeOptimizer):
    """
    The original version of: Spotted Hyena Optimizer (SHO)

    Links:
        1. https://doi.org/10.1016/j.advengsoft.2017.05.014

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + h_factor (float): default = 5, coefficient linearly decreased from 5 to 0
        + n_trials (int): default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import SHO    >>> import numpy as np
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
    >>> model = SHO.OriginalSHO(epoch=1000, pop_size=50, h_factor = 5.0, n_trials = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Dhiman, G. and Kumar, V., 2017. Spotted hyena optimizer: a novel bio-inspired based metaheuristic
    technique for engineering applications. Advances in Engineering Software, 114, pp.48-70.
    """

    cdef public object h_factor
    cdef public object n_trials

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        h_factor: float = 5.0,
        n_trials: int = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            h_factor (float): default = 5, coefficient linearly decreased from 5.0 to 0
            n_trials (int): default = 10,
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "h_factor", "n_trials"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.h_factor = cy.validator(float, h_factor, (0.5, 10.0), "h_factor")
        self.n_trials = cy.validator(int, n_trials, (1, float("inf")), "n_trials")

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativePopulation pop = self.pop
        cdef NativePopulation trial
        cdef Py_ssize_t t, n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = pop.X
        g = np.array(self.g_best_x())
        gb_fit = self.current_g_best().target.fitness
        lb, ub = self.problem.lb, self.problem.ub
        hh = self.h_factor - epoch * (self.h_factor / self.epoch)
        B = 2 * rng.uniform(0, 1, (n, d))
        E = 2 * hh * rng.uniform(0, 1, (n, d)) - hh
        bg = B @ g  # np.dot(B, g_best) of every agent
        # exploitation branch: encircle the best (a scalar step per agent, as np.dot(E, D_h))
        D_h = np.abs(bg[:, None] - X)
        pos = g - np.einsum("ij,ij->i", E, D_h)[:, None]
        # exploration branch: try to find better positions around the best, then average a circle of hunters
        hunt = np.flatnonzero(rng.random(n) >= 0.5)
        m = len(hunt)
        if m:
            found = np.zeros(m, dtype=bool)
            count = np.ones(m, dtype=int)  # N in the classic code
            trial = pop.take(np.zeros(m, dtype=int))
            for t in range(self.n_trials):
                active = np.flatnonzero(~found)
                if not len(active):
                    break
                sub = trial.take(np.arange(len(active)))
                sub.X[:] = self.correct_solution(g + rng.normal(0, 1, (len(active), d)) * rng.uniform(lb, ub, (len(active), d)))
                self.evaluate(sub, 0, len(active))
                ok = ops.better(self, sub.F, gb_fit)
                count[active] += 1
                found[active[ok]] = True
            # circle of N distinct agents per hunter
            count = np.minimum(count, n)
            Nmax = int(count.max())
            order = rng.random((m, n)).argsort(axis=1)[:, :Nmax]
            S = np.einsum("ij,ikj->ik", E[hunt], np.abs(bg[hunt][:, None, None] - X[order]))
            mask = np.arange(Nmax)[None, :] < count[:, None]
            pos[hunt] = g - ((S * mask).sum(axis=1) / count)[:, None]
        ops.step(self, pos)
