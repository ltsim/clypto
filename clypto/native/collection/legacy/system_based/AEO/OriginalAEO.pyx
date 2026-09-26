#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class OriginalAEO(_LegacyOptimizer):
    """
    The original version of: Artificial Ecosystem-based Optimization (AEO)

    Links:
        1. https://doi.org/10.1007/s00521-019-04452-x
        2. https://www.mathworks.com/matlabcentral/fileexchange/72685-artificial-ecosystem-based-optimization-aeo

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.OriginalAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2020. Artificial ecosystem-based optimization: a novel
    nature-inspired meta-heuristic algorithm. Neural Computing and Applications, 32(13), pp.9383-9425.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = True

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production   - Update the worst agent
        # Eq. 2, 3, 1
        a = (1.0 - epoch / self.epoch) * self.generator.uniform()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(
            self.problem.lb, self.problem.ub
        )
        pos_new = self.correct_solution(x1)
        agent = self.generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            jdx = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand < 1.0 / 3:
                x_t1 = self.pop[idx].solution + c * (
                        self.pop[idx].solution - self.pop[0].solution
                )  # Eq. 6
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:
                x_t1 = self.pop[idx].solution + c * (
                        self.pop[idx].solution - self.pop[jdx].solution
                )  # Eq. 7
            ### Omnivore
            else:
                r2 = self.generator.uniform()
                x_t1 = self.pop[idx].solution + c * (
                        r2 * (self.pop[idx].solution - self.pop[0].solution)
                        + (1 - r2) * (self.pop[idx].solution - self.pop[jdx].solution)
                )
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.update_target_for_population(pop_new)
            self.pop[:-1] = self.greedy_selection_population(
                self.pop[:-1], pop_new, self.problem.minmax
            )
        ## find current best used in decomposition
        best = self.get_best_agent(self.pop, self.problem.minmax)
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            x_t1 = best.solution + d * (e * best.solution - h * self.pop[idx].solution)
            pos_new = self.correct_solution(x_t1)
            agent = self.generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[idx] = self.get_better_agent(
                    agent, self.pop[idx], self.problem.minmax
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.update_target_for_population(pop_child)
            self.pop = self.greedy_selection_population(
                self.pop, pop_child, self.problem.minmax
            )
