#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget
from clypto.optimizer.native.agent cimport LegacyNativeAgent


cdef class OriginalSA(LegacyNativeOptimizer):
    """
    The original version of: Simulated Annealing (SA)

    Notes:
        + SA is single-based solution, so the pop_size parameter is not matter in this algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + temp_init (float): [1, 10000], initial temperature, default=100
        + step_size (float): the step size of random movement, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import SA    >>> import numpy as np
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
    >>> model = SA.OriginalSA(epoch=1000, pop_size=50, temp_init = 100, step_size = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
    """

    cdef public object temp_init
    cdef public object step_size
    cdef public object agent_current

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 2,
        temp_init: float = 100,
        step_size: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            temp_init (float): initial temperature, default=100
            step_size (float): the step size of random movement, default=0.1
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "temp_init", "step_size"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [2, 10000], "pop_size")
        self.temp_init = cy.validator(float, temp_init, [1, 10000], "temp_init")
        self.step_size = cy.validator(float, step_size, (-100.0, 100.0), "step_size")

    cdef void before_main_loop(self):
        self.agent_current = self.g_best.copy()

    cdef void evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef NativeTarget tar
        cdef NativePopulation pop
        cdef object cur = self.agent_current
        # Perturb the current solution
        pos_new = cur.solution + self.generator.standard_normal(self.problem.n_dims) * self.step_size
        tar = self.get_target(pos_new)
        agent = LegacyNativeAgent(pos_new, tar)
        # Accept or reject the new solution
        if self.compare_fitness(tar.fitness, cur.target.fitness, self.problem.minmax):
            self.agent_current = agent
        else:
            # Calculate the energy difference
            delta_energy = np.abs(cur.target.fitness - tar.fitness)
            # calculate probability acceptance criterion
            p_accept = np.exp(-delta_energy / (self.temp_init / epoch))
            if self.generator.random() < p_accept:
                self.agent_current = agent
        pop = self.pop.take(np.zeros(2, dtype=int))
        ops.set_row(pop, 0, self.g_best.solution, self.g_best.target)
        ops.set_row(pop, 1, self.agent_current.solution, self.agent_current.target)
        self.pop = pop
