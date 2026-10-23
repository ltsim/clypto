#!/usr/bin/env python
# Created by "Thieu" at 09:17, 09/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.human_based.BRO.DevBRO cimport DevBRO


cdef class OriginalBRO(DevBRO):
    """
    The original version of: Battle Royale Optimization (BRO)

    Links:
        1. https://doi.org/10.1007/s00521-020-05004-4

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import BRO    >>> import numpy as np
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
    >>> model = BRO.OriginalBRO(epoch=1000, pop_size=50, threshold = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Rahkar Farshi, T., 2021. Battle royale optimization algorithm. Neural Computing and Applications, 33(4), pp.1139-1157.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        threshold: float = 3,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(epoch, pop_size, threshold, **kwargs)
        self.sort_flag = False

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx, agent in enumerate(self.population.toarray()):
            # Compare ith soldier with nearest one (jth)
            jdx = self.find_idx_min_distance__(agent.solution, self.population)
            dam, vic = (
                idx,
                jdx,
            )  ## This error in the algorithm's flow in the paper, But in the matlab code, he changed.
            if cy.is_better(agent, self.population[jdx], self.problem.sense):
                dam, vic = jdx, idx  ## The mistake also here in the paper.
            if self.population[dam].damage < self.threshold:
                x = self.generator.uniform(0, 1, self.problem.n_dims) * (
                    np.maximum(self.population[dam].solution, self.g_best.solution)
                    - np.minimum(self.population[dam].solution, self.g_best.solution)
                ) + np.maximum(self.population[dam].solution, self.g_best.solution)
                x = cy.correct_solution(self.problem, x)
                child = self.population.generate_agent(x)
                child.damage = self.population[dam].damage + 1
                self.population[dam] = child
                self.population[vic].damage = 0
            else:
                x = self.generator.uniform(
                    self.lb_updated, self.ub_updated
                )
                child = self.population.generate_agent(x)
                self.population[dam] = child
        if epoch >= self.dyn_delta:
            pos_list = np.array(
                [self.population[idx].solution for idx in range(0, pop_size)]
            )
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best.solution - pos_std
            ub = self.g_best.solution + pos_std
            self.lb_updated = np.clip(
                lb, self.lb_updated, self.ub_updated
            )
            self.ub_updated = np.clip(
                ub, self.lb_updated, self.ub_updated
            )
            self.dyn_delta += round(self.dyn_delta / 2)
