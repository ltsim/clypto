#!/usr/bin/env python
# Created by "Thieu" at 00:08, 27/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalGJO(LegacyOptimizer):
    """
    The original version of: Golden jackal optimization (GJO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S095741742200358X
        2. https://www.mathworks.com/matlabcentral/fileexchange/108889-golden-jackal-optimization-algorithm

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import GJO    >>> import numpy as np
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
    >>> model = GJO.OriginalGJO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Chopra, N., & Ansari, M. M. (2022). Golden jackal optimization: A novel nature-inspired
    optimizer for engineering applications. Expert Systems with Applications, 198, 116924.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self._set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        E1 = 1.5 * (1.0 - (epoch / self.epoch))
        RL = self._get_levy_flight_step(
            beta=1.5,
            multiplier=0.05,
            size=(self.pop_size, self.problem.n_dims),
            case=-1,
        )
        _, (male, female), _ = self._get_special_agents(
            self.pop, n_best=2, n_worst=1, sense=self.problem.sense
        )
        pop_new = []
        for idx in range(0, self.pop_size):
            male_pos = male.solution.copy()
            female_pos = female.solution.copy()
            for jdx in range(0, self.problem.n_dims):
                r1 = self.generator.random()
                E0 = 2 * r1 - 1
                E = E1 * E0
                if np.abs(E) < 1:  # EXPLOITATION
                    t1 = np.abs(
                        (
                                RL[idx, jdx] * male.solution[jdx]
                                - self.pop[idx].solution[jdx]
                        )
                    )
                    male_pos[jdx] = male.solution[jdx] - E * t1
                    t2 = np.abs(
                        (
                                RL[idx, jdx] * female.solution[jdx]
                                - self.pop[idx].solution[jdx]
                        )
                    )
                    female_pos[jdx] = female.solution[jdx] - E * t2
                else:  # EXPLORATION
                    t1 = np.abs(
                        (
                                male.solution[jdx]
                                - RL[idx, jdx] * self.pop[idx].solution[jdx]
                        )
                    )
                    male_pos[jdx] = male.solution[jdx] - E * t1
                    t2 = np.abs(
                        (
                                female.solution[jdx]
                                - RL[idx, jdx] * self.pop[idx].solution[jdx]
                        )
                    )
                    female_pos[jdx] = female.solution[jdx] - E * t2
            pos_new = (male_pos + female_pos) / 2
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = agent
        if self.mode in self.AVAILABLE_MODES:
            self.pop = self._update_target_for_population(pop_new)
