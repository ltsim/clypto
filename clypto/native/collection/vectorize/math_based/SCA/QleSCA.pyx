#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.native.collection.vectorize.math_based.SCA.DevSCA cimport DevSCA


class QTable:
    def __init__(self, n_states, n_actions, generator):
        self.n_states = n_states
        self.n_actions = n_actions
        self.generator = generator
        # Initialize the Q-table with zeros
        self.table = np.zeros((n_states, n_actions))
        # Define the ranges for r1 and r3
        self.r1_ranges = [(0, 0.666), (0.667, 1.332), (1.333, 2)]
        self.r3_ranges = [(0, 0.666), (0.667, 1.332), (1.333, 2)]
        # Define the ranges for density and distance
        self.density_ranges = [(0, 0.333), (0.334, 0.666), (0.667, 1)]
        self.distance_ranges = [(0, 0.333), (0.334, 0.666), (0.667, 1)]
        self.epsilon = 0.1

    def get_state(self, density, distance):
        density_range = next(
            i for i, r in enumerate(self.density_ranges) if density <= r[1]
        )
        distance_range = next(
            i for i, r in enumerate(self.distance_ranges) if distance <= r[1]
        )
        return density_range * 3 + distance_range

    def get_action(self, state):
        acts = self.table[state, :]
        # Find the maximum value in the array
        max_val = np.max(acts)
        # Create a boolean mask that identifies all elements with the maximum value
        max_indices = np.where(acts == max_val)[0]
        # Use np.random.choice to randomly select an index from the list of indices with maximum value
        return self.generator.choice(max_indices)

    def get_action_params(self, action):
        r1_range = self.r1_ranges[action // 3]
        r3_range = self.r3_ranges[action % 3]
        return r1_range, r3_range

    def update(self, state, action, reward, alpha=0.1, gama=0.9):
        self.table[state][action] += alpha * (
            reward + gama * np.max(self.table[state]) - self.table[state][action]
        )
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class QleSCA(DevSCA):
    """
    The original version of: QLE Sine Cosine Algorithm (QLE-SCA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417421017048

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (float): [0.1-1.0], the is the learning rate in Q-learning, default=0.1
        + gama (float): [0.1-1.0]: the discount factor, default=0.9

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.math_based import SCA    >>> import numpy as np
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
    >>> model = SCA.QleSCA(epoch=1000, pop_size=50, alpha=0.1, gama=0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Hamad, Q. S., Samma, H., Suandi, S. A., & Mohamad-Saleh, J. (2022). Q-learning embedded sine cosine
    algorithm (QLESCA). Expert Systems with Applications, 193, 116417.
    """

    cdef public double alpha
    cdef public double gama
    cdef public object q_tables

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.1,
        gama: float = 0.9,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the learning rate, default=0.1
            gama (float): the discount factor, default=0.9
        """
        super().__init__(epoch, pop_size, name=name, mode=mode)
        self._params_name_ordered = tuple(["epoch", "pop_size", "alpha", "gama"])
        self.sort_flag = False
        self.is_parallelizable = False
        self.alpha = cy.validator(float, alpha, [0.0, 1.0], "alpha")
        self.gama = cy.validator(float, gama, [0.0, 1.0], "gama")

    cdef void init_fields(self, NativePopulation pop):
        # one Q-table per agent (row), as the classic agents carried theirs
        self.q_tables = [QTable(n_states=9, n_actions=9, generator=self.generator) for _ in range(pop.n)]

    cdef object amend_solution(self, object solution):
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(
            np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub),
            solution,
            rand_pos,
        )

    def density__(self, X):
        agents = np.array(X)
        # calculate the mean of each dimension of the agents
        Y = np.mean(agents, axis=0)
        # calculate the longest diagonal length L
        distances = np.sqrt(np.sum((agents[:, np.newaxis, :] - agents) ** 2, axis=-1))
        L = np.max(distances)
        # calculate the density
        return 1 / (len(agents) * L) * np.sum(np.sqrt(np.sum((agents - Y) ** 2, axis=1)))

    def distance__(self, best, X, lb, ub):
        agents = np.array(X)
        # calculate the numerator of the distance
        numerator = np.sum(np.sqrt(np.sum((best - agents) ** 2, axis=1)))
        # calculate the denominator of the distance
        denominator = np.sum([np.sqrt(np.sum((ub - lb) ** 2)) for _ in range(0, len(agents))])
        # calculate the distance
        return numerator / denominator

    cdef void evolve(self, int epoch):
        # Each agent moves after seeing the population updated so far: sequential on rows.
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        cdef Py_ssize_t idx
        Xp = pop.X
        g_best = np.array(self.g_best_x())
        for idx in range(0, self.pop_size):
            ## Step 3: State computation
            den = self.density__(Xp)
            dis = self.distance__(g_best, Xp, self.problem.lb, self.problem.ub)
            ## Step 4: Action execution
            q_table = self.q_tables[idx]
            state = q_table.get_state(density=den, distance=dis)
            action = q_table.get_action(state=state)
            r1_bound, r3_bound = q_table.get_action_params(action)
            r1 = self.generator.uniform(r1_bound[0], r1_bound[1])
            r3 = self.generator.uniform(r3_bound[0], r3_bound[1])
            r2 = 2 * np.pi * self.generator.uniform()
            r4 = self.generator.uniform()
            if r4 < 0.5:
                pos_new = Xp[idx] + r1 * np.sin(r2) * (r3 * g_best - Xp[idx])
            else:
                pos_new = Xp[idx] + r1 * np.cos(r2) * (r3 * g_best - Xp[idx])
            # Check the bound
            pos_new = self.correct_solution(pos_new)
            tar = self.get_target(pos_new)
            if self.compare_fitness(tar.fitness, pop.F[idx], self.problem.minmax):
                ops.set_row(pop, idx, pos_new, tar)
                q_table.update(state, action, reward=1, alpha=self.alpha, gama=self.gama)
            else:
                q_table.update(state, action, reward=-1, alpha=self.alpha, gama=self.gama)
