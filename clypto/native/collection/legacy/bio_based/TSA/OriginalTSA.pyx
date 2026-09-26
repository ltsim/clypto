#!/usr/bin/env python
# Created by "Thieu" at 17:23, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalTSA(LegacyOptimizer):
    """
    The original version: Tunicate Swarm Algorithm (TSA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0952197620300385?via%3Dihub
        2. https://www.mathworks.com/matlabcentral/fileexchange/75182-tunicate-swarm-algorithm-tsa

    Notes:
        1. This algorithm has some limitations
        2. The paper has several wrong equations in algorithm
        3. The implementation in Matlab code has some difference to the paper
        4. This algorithm shares some similarities with the Barnacles Mating Optimizer (BMO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import TSA    >>> import numpy as np
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
    >>> model = TSA.OriginalTSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Kaur, S., Awasthi, L. K., Sangal, A. L., & Dhiman, G. (2020). Tunicate Swarm Algorithm: A new bio-inspired
    based metaheuristic paradigm for global optimization. Engineering Applications of Artificial Intelligence, 90, 103541.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
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
        pmin, pmax = 1, 4
        pop_new = []
        for idx in range(0, self.pop_size):
            c3 = self.generator.random(self.problem.n_dims)
            c2 = self.generator.random(self.problem.n_dims)
            c1 = self.generator.random(self.problem.n_dims)
            M = np.fix(pmin + self.generator.random() * (pmax - pmin))
            A = (c2 + c3 - 2 * c1) / M
            t1 = self.g_best.solution + A * np.abs(
                self.g_best.solution - c2 * self.pop[idx].solution
            )
            t2 = self.g_best.solution - A * np.abs(
                self.g_best.solution - c2 * self.pop[idx].solution
            )
            pos_new = np.where(c3 >= 0.5, t1, t2)
            if idx != 0:
                pos_new = (pos_new + self.pop[idx - 1].solution) / 2
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self._get_target(pos_new)
        self.pop = self._update_target_for_population(pop_new)
