#!/usr/bin/env python
# Created by "Thieu" at 22:08, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class SwarmSA(LegacyOptimizer):
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
    >>> from clypto.native.collection.legacy.physics_based import SA    >>> import numpy as np
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
            **kwargs: object
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
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.max_sub_iter = self.validator.check_int(
            "max_sub_iter", max_sub_iter, [1, 100000]
        )
        self.t0 = self.validator.check_int("t0", t0, [500, 2000])
        self.t1 = self.validator.check_int("t1", t1, [1, 100])
        self.move_count = self.validator.check_int(
            "move_count", move_count, [2, int(self.pop_size / 2)]
        )
        self.mutation_rate = self.validator.check_float(
            "mutation_rate", mutation_rate, (0, 1.0)
        )
        self.mutation_step_size = self.validator.check_float(
            "mutation_step_size", mutation_step_size, (0, 1.0)
        )
        self.mutation_step_size_damp = self.validator.check_float(
            "mutation_step_size_damp", mutation_step_size_damp, (0, 1.0)
        )
        self._set_parameters(
            [
                "epoch",
                "pop_size",
                "max_sub_iter",
                "t0",
                "t1",
                "move_count",
                "mutation_rate",
                "mutation_step_size",
                "mutation_step_size_damp",
            ]
        )
        self.sort_flag = True
        self.dyn_t, self.t_damp, self.dyn_sigma = None, None, None

    def mutate__(self, position, sigma):
        # Select Mutating Variables
        pos_new = position + sigma * self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up
        )
        pos_new = np.where(
            self.generator.random(self.problem.n_dims) < self.mutation_rate,
            position,
            pos_new,
        )
        if np.all(pos_new == position):  # Select at least one variable to mutate
            pos_new[self.generator.integers(0, self.problem.n_dims)] = (
                self.generator.uniform()
            )
        return self._correct_solution(pos_new)

    def _initialization(self):
        # Initial Temperature
        self.dyn_t = self.t0  # Initial Temperature
        self.t_damp = (self.t1 / self.t0) ** (
                1.0 / self.epoch
        )  # Calculate Temperature Damp Rate
        self.dyn_sigma = self.mutation_step_size  # Initial Value of Step Size
        if self.pop is None:
            self.pop = self._generate_population(self.pop_size)

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Sub-Iterations
        for g in range(0, self.max_sub_iter):
            # Create new population
            pop_new = []
            for idx in range(0, self.pop_size):
                for j in range(0, self.move_count):
                    # Perform Mutation (Move)
                    pos_new = self.mutate__(self.pop[idx].solution, self.dyn_sigma)
                    pos_new = self._correct_solution(pos_new)
                    agent = self._generate_empty_agent(pos_new)
                    pop_new.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_new[-1].target = self._get_target(pos_new)
            pop_new = self._update_target_for_population(pop_new)
            # Columnize and Sort Newly Created Population
            pop_new = self._get_sorted_and_trimmed_population(
                pop_new, self.pop_size, self.problem.sense
            )
            # Randomized Selection
            for idx in range(0, self.pop_size):
                # Check if new solution is better than current
                if self._compare_target(
                        pop_new[idx].target, self.pop[idx].target, self.problem.sense
                ):
                    self.pop[idx] = pop_new[idx].copy()
                else:
                    # Compute difference according to problem type
                    delta = np.abs(
                        pop_new[idx].target.fitness - self.pop[idx].target.fitness
                    )
                    p = np.exp(-delta / self.dyn_t)  # Compute Acceptance Probability
                    if self.generator.uniform() <= p:  # Accept / Reject
                        self.pop[idx] = pop_new[idx].copy()
        # Update Temperature
        self.dyn_t = self.t_damp * self.dyn_t
        self.dyn_sigma = self.mutation_step_size_damp * self.dyn_sigma
