#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalMRFO(LegacyOptimizer):
    """
    The original version of: Manta Ray Foraging Optimization (MRFO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103300

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + somersault_range (float): [1.5, 3], somersault factor that decides the somersault range of manta rays, default=2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import MRFO    >>> import numpy as np
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
    >>> model = MRFO.OriginalMRFO(epoch=1000, pop_size=50, somersault_range = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Zhang, Z. and Wang, L., 2020. Manta ray foraging optimization: An effective bio-inspired
    optimizer for engineering applications. Engineering Applications of Artificial Intelligence, 87, p.103300.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            somersault_range: float = 2.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            somersault_range (float): somersault factor that decides the somersault range of manta rays, default=2
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.somersault_range = self.validator.check_float(
            "somersault_range", somersault_range, [1.0, 5.0]
        )
        self._set_parameters(["epoch", "pop_size", "somersault_range"])
        self.sort_flag = False

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            # Cyclone foraging (Eq. 5, 6, 7)
            if self.generator.random() < 0.5:
                r1 = self.generator.uniform()
                beta = (
                        2
                        * np.exp(r1 * (self.epoch - epoch) / self.epoch)
                        * np.sin(2 * np.pi * r1)
                )

                if (epoch + 1) / self.epoch < self.generator.random():
                    x_rand = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
                    if idx == 0:
                        x_t1 = (
                                x_rand
                                + self.generator.uniform()
                                * (x_rand - self.pop[idx].solution)
                                + beta * (x_rand - self.pop[idx].solution)
                        )
                    else:
                        x_t1 = (
                                x_rand
                                + self.generator.uniform()
                                * (self.pop[idx - 1].solution - self.pop[idx].solution)
                                + beta * (x_rand - self.pop[idx].solution)
                        )
                else:
                    if idx == 0:
                        x_t1 = (
                                self.g_best.solution
                                + self.generator.uniform()
                                * (self.g_best.solution - self.pop[idx].solution)
                                + beta * (self.g_best.solution - self.pop[idx].solution)
                        )
                    else:
                        x_t1 = (
                                self.g_best.solution
                                + self.generator.uniform()
                                * (self.pop[idx - 1].solution - self.pop[idx].solution)
                                + beta * (self.g_best.solution - self.pop[idx].solution)
                        )
            # Chain foraging (Eq. 1,2)
            else:
                r = self.generator.uniform()
                alpha = 2 * r * np.sqrt(np.abs(np.log(r)))
                if idx == 0:
                    x_t1 = (
                            self.pop[idx].solution
                            + r * (self.g_best.solution - self.pop[idx].solution)
                            + alpha * (self.g_best.solution - self.pop[idx].solution)
                    )
                else:
                    x_t1 = (
                            self.pop[idx].solution
                            + r * (self.pop[idx - 1].solution - self.pop[idx].solution)
                            + alpha * (self.g_best.solution - self.pop[idx].solution)
                    )
            pos_new = self._correct_solution(x_t1)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    self.pop[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            self.pop = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )
        _, g_best = self._update_global_best_agent(self.pop)
        pop_child = []
        for idx in range(0, self.pop_size):
            # Somersault foraging   (Eq. 8)
            x_t1 = self.pop[idx].solution + self.somersault_range * (
                    self.generator.uniform() * g_best.solution
                    - self.generator.uniform() * self.pop[idx].solution
            )
            pos_new = self._correct_solution(x_t1)
            agent = self._generate_empty_agent(pos_new)
            pop_child.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                self.pop[idx] = self._get_better_agent(
                    self.pop[idx], agent, self.problem.sense
                )
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self._update_target_for_population(pop_child)
            self.pop = self._greedy_selection_population(
                self.pop, pop_child, self.problem.sense
            )
