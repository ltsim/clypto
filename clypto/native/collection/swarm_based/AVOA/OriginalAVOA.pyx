#!/usr/bin/env python
# Created by "Thieu" at 21:45, 26/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalAVOA(cy.Optimizer):
    """
    The original version of: African Vultures Optimization Algorithm (AVOA)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0360835221003120
        2. https://www.mathworks.com/matlabcentral/fileexchange/94820-african-vultures-optimization-algorithm

    Notes (parameters):
        + p1 (float): probability of status transition, default 0.6
        + p2 (float): probability of status transition, default 0.4
        + p3 (float): probability of status transition, default 0.6
        + alpha (float): probability of 1st best, default = 0.8
        + gama (float): a factor in the paper (not much affect to algorithm), default = 2.5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import AVOA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = AVOA.OriginalAVOA(epoch=1000, pop_size=50, p1=0.6, p2=0.4, p3=0.6, alpha=0.8, gama=2.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new
    nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.
    """

    cdef public double alpha
    cdef public double gama
    cdef public double p1
    cdef public double p2
    cdef public double p3

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            p1: float = 0.6,
            p2: float = 0.4,
            p3: float = 0.6,
            alpha: float = 0.8,
            gama: float = 2.5,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size", "p1", "p2", "p3", "alpha", "gama"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.p1 = cy.validator(float, p1, (0, 1), "p1")
        self.p2 = cy.validator(float, p2, (0, 1), "p2")
        self.p3 = cy.validator(float, p3, (0, 1), "p3")
        self.alpha = cy.validator(float, alpha, (0, 1), "alpha")
        self.gama = cy.validator(float, gama, (0, 5.0), "gama")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        a = self.generator.uniform(-2, 2) * (
                (np.sin((np.pi / 2) * (epoch / self.epoch)) ** self.gama)
                + np.cos((np.pi / 2) * (epoch / self.epoch))
                - 1
        )
        ppp = (2 * self.generator.random() + 1) * (1 - epoch / self.epoch) + a
        ranked = self.population.sort()
        best_list = [cy.duplicate_agent(agent) for agent in ranked[:2]]
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            F = ppp * (2 * self.generator.random() - 1)
            rand_idx = self.generator.choice([0, 1], p=[self.alpha, 1 - self.alpha])
            rand_pos = best_list[rand_idx].solution
            if np.abs(F) >= 1:  # Exploration
                if self.generator.random() < self.p1:
                    x = (
                            rand_pos
                            - (
                                np.abs(
                                    (2 * self.generator.random()) * rand_pos
                                    - agent.solution
                                )
                            )
                            * F
                    )
                else:
                    x = (
                            rand_pos
                            - F
                            + self.generator.random()
                            * (
                                    (self.problem.bounds.up - self.problem.bounds.low)
                                    * self.generator.random()
                                    + self.problem.bounds.low
                            )
                    )
            else:  # Exploitation
                if np.abs(F) < 0.5:  # Phase 1
                    best_x1 = best_list[0].solution
                    best_x2 = best_list[1].solution
                    if self.generator.random() < self.p2:
                        A = (
                                best_x1
                                - (
                                        (best_x1 * agent.solution)
                                        / (best_x1 - agent.solution ** 2 + self.EPSILON)
                                )
                                * F
                        )
                        B = (
                                best_x2
                                - (
                                        (best_x2 * agent.solution)
                                        / (best_x2 - agent.solution ** 2 + self.EPSILON)
                                )
                                * F
                        )
                        x = (A + B) / 2
                    else:
                        x = rand_pos - np.abs(
                            rand_pos - agent.solution
                        ) * F * cy.levy_flight(self.generator, beta=1.5, multiplier=1.0, size=self.problem.n_dims, case=-1)
                else:  # Phase 2
                    if self.generator.random() < self.p3:
                        x = (
                                      np.abs(
                                          (2 * self.generator.random()) * rand_pos
                                          - agent.solution
                                      )
                                  ) * (F + self.generator.random()) - (
                                          rand_pos - agent.solution
                                  )
                    else:
                        s1 = (
                                rand_pos
                                * (
                                        self.generator.random()
                                        * agent.solution
                                        / (2 * np.pi)
                                )
                                * np.cos(agent.solution)
                        )
                        s2 = (
                                rand_pos
                                * (
                                        self.generator.random()
                                        * agent.solution
                                        / (2 * np.pi)
                                )
                                * np.sin(agent.solution)
                        )
                        x = rand_pos - (s1 + s2)
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))
