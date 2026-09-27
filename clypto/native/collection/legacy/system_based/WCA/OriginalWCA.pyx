#!/usr/bin/env python
# Created by "Thieu" at 09:55, 02/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalWCA(cy.Optimizer):
    """
    The original version of: Water Cycle Algorithm (WCA)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2012.07.010

    Notes
    ~~~~~
    The ideas are (almost the same as ICO algorithm):
        + 1 sea is global best solution
        + a few river which are second, third, ...
        + other left are stream (will flow directed to sea or river)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + nsr (int): [4, 10], Number of rivers + sea (sea = 1), default = 4
        + wc (float): [1.0, 3.0], Weighting coefficient (C in the paper), default = 2
        + dmax (float): [1e-6], fixed parameter, Evaporation condition constant, default=1e-6

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.system_based import WCA    >>> import numpy as np
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
    >>> model = WCA.OriginalWCA(epoch=1000, pop_size=50, nsr = 4, wc = 2.0, dmax = 1e-6)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Eskandar, H., Sadollah, A., Bahreininejad, A. and Hamdi, M., 2012. Water cycle algorithm–A novel metaheuristic
    optimization method for solving constrained engineering optimization problems. Computers & Structures, 110, pp.151-166.
    """

    cdef public double dmax
    cdef public int nsr
    cdef public double wc

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            nsr: int = 4,
            wc: float = 2.0,
            dmax: float = 1e-6,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            nsr (int): Number of rivers + sea (sea = 1), default = 4
            wc (float): Weighting coefficient (C in the paper), default = 2.0
            dmax (float): Evaporation condition constant, default=1e-6
        """
        super().__init__(parameters=["epoch", "pop_size", "nsr", "wc", "dmax"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.nsr = cy.validator(int, nsr, [2, int(self.population.size() / 2)], "nsr")
        self.wc = cy.validator(float, wc, (1.0, 3.0), "wc")
        self.dmax = cy.validator(float, dmax, (0, 1.0), "dmax")

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.generate(pop_size)
        self.population = self.population.sort()
        self.g_best = self.population[0]
        self.ecc = self.dmax  # Evaporation condition constant - variable
        n_stream = pop_size - self.nsr
        g_best = self.population[0].copy()  # Global best solution (sea)
        self.pop_best = self.population[
            : self.nsr
        ]  # Including sea and river (1st solution is sea)
        self.pop_stream = self.population[self.nsr:]  # Forming Stream

        # Designate streams to rivers and sea
        cost_river_list = np.array([agent.fitness for agent in self.pop_best])
        num_child_in_river_list = np.round(
            np.abs(cost_river_list / np.sum(cost_river_list)) * n_stream
        ).astype(int)
        if np.sum(num_child_in_river_list) < n_stream:
            num_child_in_river_list[-1] += n_stream - np.sum(num_child_in_river_list)
        streams = {}
        idx_already_selected = []
        for i in range(0, self.nsr - 1):
            streams[i] = []
            idx_list = self.generator.choice(
                list(set(range(0, n_stream)) - set(idx_already_selected)),
                num_child_in_river_list[i],
                replace=False,
            ).tolist()
            idx_already_selected += idx_list
            for idx in idx_list:
                streams[i].append(self.pop_stream[idx])
        idx_last = list(set(range(0, n_stream)) - set(idx_already_selected))
        streams[self.nsr - 1] = []
        for idx in idx_last:
            streams[self.nsr - 1].append(self.pop_stream[idx])
        self.streams = streams

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Update stream and river
        for idx, stream_list in self.streams.items():
            # Update stream
            stream_new = []
            for idx_stream, stream in enumerate(stream_list):
                pos_new = stream.solution + self.generator.uniform() * self.wc * (
                        self.pop_best[idx].solution - stream.solution
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.create_agent(pos_new)
                stream_new.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    stream_new[-1].evaluate(self.problem)
            stream_new = self.population.evaluate(stream_new, self.mode)
            self.streams[idx] = stream_new
            stream_best = cy.sort_agents(stream_new, self.problem.sense)[0].copy()
            if cy.is_better(stream_best, self.pop_best[idx], self.problem.sense):
                self.pop_best[idx] = stream_best.copy()
            # Update river
            pos_new = self.pop_best[
                          idx
                      ].solution + self.generator.uniform() * self.wc * (
                              self.g_best.solution - self.pop_best[idx].solution
                      )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.generate_agent(pos_new)
            if cy.is_better(agent, self.pop_best[idx], self.problem.sense):
                self.pop_best[idx] = agent
        # Evaporation
        for idx in range(1, self.nsr):
            distance = np.sqrt(
                np.sum((self.g_best.solution - self.pop_best[idx].solution) ** 2)
            )
            if distance < self.ecc or self.generator.random() < 0.1:
                child = self.population.generate_agent()
                pop_current_best = cy.sort_agents(self.streams[idx] + [child], self.problem.sense)
                self.pop_best[idx] = pop_current_best.pop(0)
                self.streams[idx] = pop_current_best
        self.population = self.pop_best.copy()
        for idx, stream_list in self.streams.items():
            self.population += stream_list
        # Reduce the ecc
        self.ecc = self.ecc - self.ecc / self.epoch
