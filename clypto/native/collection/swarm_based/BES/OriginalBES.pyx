#!/usr/bin/env python
# Created by "Thieu" at 14:52, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalBES(cy.Optimizer):
    """
    The original version of: Bald Eagle Search (BES)

    Links:
        1. https://doi.org/10.1007/s10462-019-09732-5

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + a_factor (int): default: 10, determining the corner between point search in the central point, in [5, 10]
        + R_factor (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
        + alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
        + c1 (float): default: 2, in [1, 2]
        + c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import BES    >>> import numpy as np
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
    >>> model = BES.OriginalBES(epoch=1000, pop_size=50, a_factor = 10, R_factor = 1.5, alpha = 2.0, c1 = 2.0, c2 = 2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Alsattar, H.A., Zaidan, A.A. and Zaidan, B.B., 2020. Novel meta-heuristic bald eagle
    search optimisation algorithm. Artificial Intelligence Review, 53(3), pp.2237-2264.
    """

    cdef public double R_factor
    cdef public int a_factor
    cdef public double alpha
    cdef public double c1
    cdef public double c2

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            a_factor: int = 10,
            R_factor: float = 1.5,
            alpha: float = 2.0,
            c1: float = 2.0,
            c2: float = 2.0,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            a_factor (int): default: 10, determining the corner between point search in the central point, in [5, 10]
            R_factor (float): default: 1.5, determining the number of search cycles, in [0.5, 2]
            alpha (float): default: 2, parameter for controlling the changes in position, in [1.5, 2]
            c1 (float): default: 2, in [1, 2]
            c2 (float): c1 and c2 increase the movement intensity of bald eagles towards the best and centre points
        """
        super().__init__(parameters=["epoch", "pop_size", "a_factor", "R_factor", "alpha", "c1", "c2"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.a_factor = cy.validator(int, a_factor, [2, 20], "a_factor")
        self.R_factor = cy.validator(float, R_factor, [0.1, 3.0], "R_factor")
        self.alpha = cy.validator(float, alpha, [0.5, 3.0], "alpha")
        self.c1 = cy.validator(float, c1, (0, 4.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 4.0), "c2")

    def create_x_y_x1_y1__(self):
        """Using numpy vector for faster computational time"""
        pop_size = self.population.size()
        ## Eq. 2
        phi = self.a_factor * np.pi * self.generator.uniform(0, 1, pop_size)
        r = phi + self.R_factor * self.generator.uniform(0, 1, pop_size)
        xr, yr = r * np.sin(phi), r * np.cos(phi)
        ## Eq. 3
        r1 = phi1 = self.a_factor * np.pi * self.generator.uniform(0, 1, pop_size)
        xr1, yr1 = r1 * np.sinh(phi1), r1 * np.cosh(phi1)
        x_list = xr / np.max(xr)
        y_list = yr / np.max(yr)
        x1_list = xr1 / np.max(xr1)
        y1_list = yr1 / np.max(yr1)
        return x_list, y_list, x1_list, y1_list

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## 0. Pre-definded
        x_list, y_list, x1_list, y1_list = self.create_x_y_x1_y1__()

        # Three parts: selecting the search space, searching within the selected search space and swooping.
        ## 1. Select space
        pos_list = np.array([agent.solution for agent in self.population])
        pos_mean = np.mean(pos_list, axis=0)

        pop_new = []
        for idx in range(0, pop_size):
            x = self.g_best.solution + self.alpha * self.generator.uniform() * (
                    pos_mean - self.population[idx].solution
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_new.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)

        ## 2. Search in space
        pos_list = np.array([agent.solution for agent in self.population])
        pos_mean = np.mean(pos_list, axis=0)
        pop_child = []
        for idx in range(0, pop_size):
            idx_rand = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            x = (
                    self.population[idx].solution
                    + y_list[idx] * (self.population[idx].solution - self.population[idx_rand].solution)
                    + x_list[idx] * (self.population[idx].solution - pos_mean)
            )
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            pop_child.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_child = self.population.evaluate(pop_child, self.mode)
            self.population = self.population.greedy(pop_child)

        ## 3. Swoop
        pos_list = np.array([agent.solution for agent in self.population])
        pos_mean = np.mean(pos_list, axis=0)
        pop_new = []
        for idx, agent in enumerate(self.population.toarray()):
            x = (
                    self.generator.uniform() * self.g_best.solution
                    + x1_list[idx] * (agent.solution - self.c1 * pos_mean)
                    + y1_list[idx]
                    * (agent.solution - self.c2 * self.g_best.solution)
            )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            pop_new.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
