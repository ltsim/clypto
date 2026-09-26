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


cdef class SwarmSA(LegacyNativeOptimizer):
    """
    The swarm version of: Simulated Annealing (SwarmSA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + max_sub_iter (int): [5, 10, 15], Maximum Number of Sub-Iteration (within fixed temperature), default=5
        + t0 (int): Fixed parameter, Initial Temperature, default=1000
        + t1 (int): Fixed parameter, Final Temperature, default=1
        + move_count (int): [5, 20], Move Count per Individual Solution, default=5
        + mutation_rate (float): [0.01, 0.2], Mutation Rate, default=0.1
        + mutation_step_size (float): [0.05, 0.1, 0.15], Mutation Step Size, default=0.1
        + mutation_step_size_damp (float): [0.8, 0.99], Mutation Step Size Damp, default=0.99

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import SA    >>> import numpy as np
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
    >>> model = SA.SwarmSA(epoch=1000, pop_size=50, max_sub_iter = 5, t0 = 1000, t1 = 1,
    >>>         move_count = 5, mutation_rate = 0.1, mutation_step_size = 0.1, mutation_step_size_damp = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Van Laarhoven, P.J. and Aarts, E.H., 1987. Simulated annealing. In Simulated
    annealing: Theory and applications (pp. 7-15). Springer, Dordrecht.
    """

    cdef public object max_sub_iter
    cdef public object t0
    cdef public object t1
    cdef public object move_count
    cdef public object mutation_rate
    cdef public object mutation_step_size
    cdef public object mutation_step_size_damp
    cdef public object dyn_sigma
    cdef public object dyn_t
    cdef public object t_damp

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        max_sub_iter: int = 5,
        t0: int = 1000,
        t1: int = 1,
        move_count: int = 5,
        mutation_rate: float = 0.1,
        mutation_step_size: float = 0.1,
        mutation_step_size_damp: float = 0.99,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            max_sub_iter (int): Maximum Number of Sub-Iteration (within fixed temperature), default=5
            t0 (int): Initial Temperature, default=1000
            t1 (int): Final Temperature, default=1
            move_count (int): Move Count per Individual Solution, default=5
            mutation_rate (float): Mutation Rate, default=0.1
            mutation_step_size (float): Mutation Step Size, default=0.1
            mutation_step_size_damp (float): Mutation Step Size Damp, default=0.99
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "max_sub_iter",
                "t0",
                "t1",
                "move_count",
                "mutation_rate",
                "mutation_step_size",
                "mutation_step_size_damp",
            ],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.max_sub_iter = cy.validator(int, max_sub_iter, [1, 100000], "max_sub_iter")
        self.t0 = cy.validator(int, t0, [500, 2000], "t0")
        self.t1 = cy.validator(int, t1, [1, 100], "t1")
        self.move_count = cy.validator(int, move_count, [2, int(self.pop_size / 2)], "move_count")
        self.mutation_rate = cy.validator(float, mutation_rate, (0, 1.0), "mutation_rate")
        self.mutation_step_size = cy.validator(float, mutation_step_size, (0, 1.0), "mutation_step_size")
        self.mutation_step_size_damp = cy.validator(float, mutation_step_size_damp, (0, 1.0), "mutation_step_size_damp")
        self.dyn_t, self.t_damp, self.dyn_sigma = None, None, None

    def mutate__(self, position, sigma):
        # Select Mutating Variables
        pos_new = position + sigma * self.generator.uniform(self.problem.lb, self.problem.ub)
        pos_new = np.where(
            self.generator.random(self.problem.n_dims) < self.mutation_rate,
            position,
            pos_new,
        )
        if np.all(pos_new == position):  # Select at least one variable to mutate
            pos_new[self.generator.integers(0, self.problem.n_dims)] = self.generator.uniform()
        return self.correct_solution(pos_new)

    cdef void initialization(self):
        # Initial Temperature
        self.dyn_t = self.t0  # Initial Temperature
        self.t_damp = (self.t1 / self.t0) ** (1.0 / self.epoch)  # Calculate Temperature Damp Rate
        self.dyn_sigma = self.mutation_step_size  # Initial Value of Step Size
        LegacyNativeOptimizer.initialization(self)

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand, best
        cdef Py_ssize_t g, idx, j, k, n = pop.n, d = pop.d
        # Sub-Iterations
        for g in range(0, self.max_sub_iter):
            # Create new population (move_count mutations per agent, drawn agent by agent)
            X, moves = pop.X, []
            for idx in range(0, self.pop_size):
                for j in range(0, self.move_count):
                    # Perform Mutation (Move)
                    moves.append(self.correct_solution(self.mutate__(X[idx], self.dyn_sigma)))
            cand = self.new_population(np.array(moves))
            # Columnize and Sort Newly Created Population
            best = cand.take(self.sorted_order(cand)[:self.pop_size])
            # Randomized Selection
            for idx in range(0, self.pop_size):
                # Check if new solution is better than current
                if self.compare_fitness(best.F[idx], pop.F[idx], self.problem.minmax):
                    pop.buf[idx] = best.buf[idx]
                else:
                    # Compute difference according to problem type
                    delta = np.abs(best.F[idx] - pop.F[idx])
                    p = np.exp(-delta / self.dyn_t)  # Compute Acceptance Probability
                    if self.generator.uniform() <= p:  # Accept / Reject
                        pop.buf[idx] = best.buf[idx]
        # Update Temperature
        self.dyn_t = self.t_damp * self.dyn_t
        self.dyn_sigma = self.mutation_step_size_damp * self.dyn_sigma
