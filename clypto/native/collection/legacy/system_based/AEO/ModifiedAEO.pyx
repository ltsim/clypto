#!/usr/bin/env python
# Created by "Thieu" at 16:44, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class ModifiedAEO(LegacyOptimizer):
    """
    The original version of: Modified Artificial Ecosystem-Based Optimization (MAEO)

    Links:
        1. https://doi.org/10.1109/ACCESS.2020.2973351

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.system_based import AEO    >>> import numpy as np
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
    >>> model = AEO.ModifiedAEO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Menesy, A.S., Sultan, H.M., Korashy, A., Banakhr, F.A., Ashmawy, M.G. and Kamel, S., 2020. Effective
    parameter extraction of different polymer electrolyte membrane fuel cell stack models using a
    modified artificial ecosystem optimization algorithm. IEEE Access, 8, pp.31892-31909.
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
        self.sort_flag = True

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        ## Production
        # Eq. 22
        H = 2 * (1 - epoch / self.epoch)
        a = (1 - epoch / self.epoch) * self.generator.random()
        x1 = (1 - a) * self.pop[-1].solution + a * self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up
        )
        pos_new = self._correct_solution(x1)
        agent = self._generate_agent(pos_new)
        self.pop[-1] = agent
        ## Consumption - Update the whole population left
        pop_new = []
        for idx in range(0, self.pop_size - 1):
            rand = self.generator.random()
            # Eq. 4, 5, 6
            v1 = self.generator.normal(0, 1)
            v2 = self.generator.normal(0, 1)
            c = 0.5 * v1 / abs(v2)  # Consumption factor
            j = 1 if idx == 0 else self.generator.integers(0, idx)
            ### Herbivore
            if rand <= 1.0 / 3:  # Eq. 23
                pos_new = self.pop[idx].solution + H * c * (
                        self.pop[idx].solution - self.pop[0].solution
                )
            ### Carnivore
            elif 1.0 / 3 <= rand and rand <= 2.0 / 3:  # Eq. 24
                pos_new = self.pop[idx].solution + H * c * (
                        self.pop[idx].solution - self.pop[j].solution
                )
            ### Omnivore
            else:  # Eq. 25
                r5 = self.generator.random()
                pos_new = self.pop[idx].solution + H * c * (
                        r5 * (self.pop[idx].solution - self.pop[0].solution)
                        + (1 - r5) * (self.pop[idx].solution - self.pop[j].solution)
                )
            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    agent, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop[:-1] = self._greedy_selection_population(
                self.pop[:-1], pop_new, self.problem.sense
            )
        ## find current best used in decomposition
        best = self._get_best_agent(self.pop, self.problem.sense)
        ## Decomposition
        ### Eq. 10, 11, 12, 9
        pop_child = []
        for idx in range(0, self.pop_size):
            r3 = self.generator.uniform()
            d = 3 * self.generator.normal(0, 1)
            e = r3 * self.generator.integers(1, 3) - 1
            h = 2 * r3 - 1
            if self.generator.random() < 0.5:
                beta = 1 - (1 - 0) * (epoch / self.epoch)  # Eq. 21
                r_idx = self.generator.choice(
                    list(set(range(0, self.pop_size)) - {idx})
                )
                x_r = self.pop[r_idx].solution
                if self.generator.random() < 0.5:
                    x_new = beta * x_r + (1 - beta) * self.pop[idx].solution
                else:
                    x_new = (1 - beta) * x_r + beta * self.pop[idx].solution
            else:
                x_new = best.solution + d * (
                        e * best.solution - h * self.pop[idx].solution
                )
            pos_new = self._correct_solution(x_new)
            agent = self._generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    agent, self.pop[idx], self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self._update_target_for_population(pop_child)
            self.pop = self._greedy_selection_population(
                self.pop, pop_child, self.problem.sense
            )
