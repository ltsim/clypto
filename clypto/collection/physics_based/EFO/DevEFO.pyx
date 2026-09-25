#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation


cdef class DevEFO(LegacyNativeOptimizer):
    """
    The developed version: Electromagnetic Field Optimization (EFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EFO    >>> import numpy as np
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
    >>> model = EFO.DevEFO(epoch=1000, pop_size=50, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """


    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r_rate: float = 0.3,
        ps_rate: float = 0.85,
        p_field: float = 0.1,
        n_field: float = 0.45,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "r_rate", "ps_rate", "p_field", "n_field"],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.r_rate = cy.validator(float, r_rate, (0, 1.0), "r_rate")
        self.ps_rate = cy.validator(float, ps_rate, (0, 1.0), "ps_rate")
        self.p_field = cy.validator(float, p_field, (0, 1.0), "p_field")
        self.n_field = cy.validator(float, n_field, (0, 1.0), "n_field")
        self.phi = (1 + np.sqrt(5)) / 2

    cdef void evolve(self, int epoch):
        # Agents read rows updated before them: sequential on the buffer rows (batched in swarm modes).
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand = pop.empty_like()
        cdef Py_ssize_t idx, n = pop.n, d = pop.d
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        for idx in range(0, self.pop_size):
            r_idx1 = self.generator.integers(
                0, int(self.pop_size * self.p_field)
            )  # top
            r_idx2 = self.generator.integers(
                int(self.pop_size * (1 - self.n_field)), self.pop_size
            )  # bottom
            r_idx3 = self.generator.integers(
                int((self.pop_size * self.p_field) + 1),
                int(self.pop_size * (1 - self.n_field)),
            )  # middle
            if self.generator.random() < self.ps_rate:
                pos_new = (
                        Xp[r_idx1]
                        + self.phi
                        * self.generator.random()
                        * (g_best - Xp[r_idx3])
                        + self.generator.random()
                        * (g_best - Xp[r_idx2])
                )
            else:
                pos_new = self.problem.generate_solution()
            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if self.generator.random() < self.r_rate:
                RI = self.generator.integers(0, d)
                pos_new[self.generator.integers(0, d)] = (
                    self.generator.uniform(self.problem.lb[RI], self.problem.ub[RI])
                )
            # checking whether the generated number is inside boundary or not
            ops.commit(self, pop, cand, idx, self.correct_solution(pos_new), swarm)
        if swarm:
            ops.finish(self, cand, 0, n)
