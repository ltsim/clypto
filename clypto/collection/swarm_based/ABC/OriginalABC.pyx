#!/usr/bin/env python
# Created by "Thieu" at 09:57, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalABC(LegacyNativeOptimizer):
    """
    The original version of: Artificial Bee Colony (ABC)

    Links:
        1. https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_limits (int): Limit of trials before abandoning a food source, default=25

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import ABC    >>> import numpy as np
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
    >>> model = ABC.OriginalABC(epoch=1000, pop_size=50, n_limits = 50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] B. Basturk, D. Karaboga, An artificial bee colony (ABC) algorithm for numeric function optimization,
    in: IEEE Swarm Intelligence Symposium 2006, May 12–14, Indianapolis, IN, USA, 2006.
    """

    cdef public object n_limits
    cdef public object trials

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_limits: int = 25,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size = onlooker bees = employed bees, default = 100
            n_limits: Limit of trials before abandoning a food source, default=25
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "n_limits"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.n_limits = cy.validator(int, n_limits, [1, 1000], "n_limits")

    cdef void initialize_variables(self):
        self.trials = np.zeros(self.pop_size)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        minmax = self.problem.minmax
        Xp = pop.X
        for idx in range(0, self.pop_size):
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = Xp[idx] + phi * (Xp[rdx] - Xp[idx])
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], minmax):
                ops.set_row(pop, idx, pos_new, tar)
                self.trials[idx] = 0
            else:
                self.trials[idx] += 1
        # Onlooker bees phase
        # Calculate the probabilities of each employed bee
        employed_fits = np.array(pop.F)
        for idx in range(0, self.pop_size):
            # Select an employed bee using roulette wheel selection
            selected_bee = self.get_index_roulette_wheel_selection(employed_fits)
            # Choose a random employed bee to generate a new solution
            rdx = self.generator.choice(list(set(range(0, self.pop_size)) - {idx, selected_bee}))
            # Generate a new solution by the equation x_{ij} = x_{ij} + phi_{ij} * (x_{tj} - x_{ij})
            phi = self.generator.uniform(low=-1, high=1, size=self.problem.n_dims)
            pos_new = Xp[selected_bee] + phi * (Xp[rdx] - Xp[selected_bee])
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[selected_bee], minmax):
                ops.set_row(pop, selected_bee, pos_new, tar)
                self.trials[selected_bee] = 0
            else:
                self.trials[selected_bee] += 1
        # Scout bees phase
        # Check the number of trials for each employed bee and abandon the food source if the limit is exceeded
        abandoned = np.where(self.trials >= self.n_limits)[0]
        for idx in abandoned:
            pos_new = self.problem.generate_solution(encoded=True)
            ops.set_row(pop, idx, pos_new, self.get_target(pos_new))
            self.trials[idx] = 0
