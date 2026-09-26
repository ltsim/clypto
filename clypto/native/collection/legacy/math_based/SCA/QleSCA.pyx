#!/usr/bin/env python
# Created by "Thieu" at 17:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent
from clypto.native.collection.legacy.math_based.SCA.DevSCA cimport DevSCA


cdef class _QleSCAAgent(_LegacyAgent):
    cdef public object q_table
    def __init__(self, solution=None, target=None, q_table=None):
        _LegacyAgent.__init__(self, solution, target)
        self.q_table = q_table
    cpdef object copy(self):
        return _QleSCAAgent(
            self.solution, None if self.target is None else self.target.copy(),
            self.q_table,
        )
    def update(self, **kwargs):
        if "q_table" in kwargs:
            self.q_table = kwargs.pop("q_table")
        _LegacyAgent.update(self, **kwargs)


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
    >>> from clypto.collection.math_based import SCA    >>> import numpy as np
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

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: float = 0.1,
        gama: float = 0.9,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (float): the learning rate, default=0.1
            gama (float): the discount factor, default=0.9
        """
        super().__init__(epoch, pop_size, **kwargs)
        self.alpha = self.validator.check_float("alpha", alpha, [0.0, 1.0])
        self.gama = self.validator.check_float("gama", gama, [0.0, 1.0])
        self.set_parameters(["epoch", "pop_size", "alpha", "gama"])
        self.sort_flag = False
        self.is_parallelizable = False

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        q_table = QTable(n_states=9, n_actions=9, generator=self.generator)
        return _QleSCAAgent(solution=solution, q_table=q_table)

    def amend_solution(self, solution: np.ndarray) -> np.ndarray:
        rand_pos = self.generator.uniform(self.problem.lb, self.problem.ub)
        return np.where(
            np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub),
            solution,
            rand_pos,
        )

    def density__(self, pop):
        agents = np.array([agent.solution for agent in pop])
        # calculate the mean of each dimension of the agents
        Y = np.mean(agents, axis=0)
        # calculate the longest diagonal length L
        distances = np.sqrt(np.sum((agents[:, np.newaxis, :] - agents) ** 2, axis=-1))
        L = np.max(distances)
        # calculate the density
        return 1 / (len(pop) * L) * np.sum(np.sqrt(np.sum((agents - Y) ** 2, axis=1)))

    def distance__(self, best, pop, lb, ub):
        agents = np.array([agent.solution for agent in pop])
        # calculate the numerator of the distance
        numerator = np.sum(np.sqrt(np.sum((best.solution - agents) ** 2, axis=1)))
        # calculate the denominator of the distance
        denominator = np.sum(
            [np.sqrt(np.sum((ub - lb) ** 2)) for _ in range(0, len(pop))]
        )
        # calculate the distance
        return numerator / denominator

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        for idx in range(0, self.pop_size):
            agent = self.pop[idx].copy()
            ## Step 3: State computation
            den = self.density__(self.pop)
            dis = self.distance__(
                self.g_best, self.pop, self.problem.lb, self.problem.ub
            )
            ## Step 4: Action execution
            state = self.pop[idx].q_table.get_state(density=den, distance=dis)
            action = self.pop[idx].q_table.get_action(state=state)
            r1_bound, r3_bound = self.pop[idx].q_table.get_action_params(action)
            r1 = self.generator.uniform(r1_bound[0], r1_bound[1])
            r3 = self.generator.uniform(r3_bound[0], r3_bound[1])
            r2 = 2 * np.pi * self.generator.uniform()
            r4 = self.generator.uniform()
            if r4 < 0.5:
                pos_new = self.pop[idx].solution + r1 * np.sin(r2) * (
                    r3 * self.g_best.solution - self.pop[idx].solution
                )
            else:
                pos_new = self.pop[idx].solution + r1 * np.cos(r2) * (
                    r3 * self.g_best.solution - self.pop[idx].solution
                )
            # Check the bound
            pos_new = self.correct_solution(pos_new)
            agent.solution = pos_new
            agent.target = self.get_target(pos_new)
            if self.compare_target(
                agent.target, self.pop[idx].target, self.problem.minmax
            ):
                self.pop[idx] = agent
                self.pop[idx].q_table.update(
                    state, action, reward=1, alpha=self.alpha, gama=self.gama
                )
            else:
                self.pop[idx].q_table.update(
                    state, action, reward=-1, alpha=self.alpha, gama=self.gama
                )
