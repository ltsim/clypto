#!/usr/bin/env python
# Created by "Thieu" at 07:50, 14/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalRUN(cy.Optimizer):
    """
    The original version of: RUNge Kutta optimizer (RUN)

    Links:
        1. https://doi.org/10.1016/j.eswa.2021.115079
        2. https://imanahmadianfar.com/codes/
        3. https://www.aliasgharheidari.com/RUN.html

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import RUN    >>> import numpy as np
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
    >>> model = RUN.OriginalRUN(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Ahmadianfar, I., Heidari, A. A., Gandomi, A. H., Chu, X., & Chen, H. (2021). RUN beyond the metaphor: An efficient
    optimization algorithm based on Runge Kutta method. Expert Systems with Applications, 181, 115079.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def runge_kutta__(self, xb, xw, delta_x):
        dim = len(xb)
        C = self.generator.integers(1, 3) * (1 - self.generator.random())
        r1 = self.generator.random(dim)
        r2 = self.generator.random(dim)
        K1 = 0.5 * (self.generator.random() * xw - C * xb)
        K2 = 0.5 * (
                self.generator.random() * (xw + r2 * K1 * delta_x / 2)
                - (C * xb + r1 * K1 * delta_x / 2)
        )
        K3 = 0.5 * (
                self.generator.random() * (xw + r2 * K2 * delta_x / 2)
                - (C * xb + r1 * K2 * delta_x / 2)
        )
        K4 = 0.5 * (
                self.generator.random() * (xw + r2 * K3 * delta_x)
                - (C * xb + r1 * K3 * delta_x)
        )
        return (K1 + 2 * K2 + 2 * K3 + K4) / 6

    def uniform_random__(self, a, b, size):
        a2, b2 = a / 2, b / 2
        mu = a2 + b2
        sig = b2 - a2
        return mu + sig * (2 * self.generator.uniform(0, 1, size) - 1)

    def get_index_of_best_agent__(self, pop):
        fit_list = np.array([agent.fitness for agent in pop])
        if self.problem.sense == "min":
            return np.argmin(fit_list)
        else:
            return np.argmax(fit_list)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        f = 20 * np.exp(-(12.0 * epoch / self.epoch))  # Eq.17.6
        SF = 2.0 * (0.5 - self.generator.random(pop_size)) * f  # Eq.17.5
        x_list = np.array([agent.solution for agent in self.population])
        x_average = np.mean(x_list, axis=0)  # Determine the Average of Solutions
        for idx, agent in enumerate(self.population.toarray()):
            ## Determine Delta X (Eqs. 11.1 to 11.3)
            gama = (
                    self.generator.random()
                    * (
                            agent.solution
                            - self.generator.uniform(0, 1, self.problem.n_dims)
                            * (self.problem.bounds.up - self.problem.bounds.low)
                    )
                    * np.exp(-4 * epoch / self.epoch)
            )
            stp = self.generator.uniform(0, 1, self.problem.n_dims) * (
                    (self.g_best.solution - self.generator.random() * x_average) + gama
            )
            delta_x = (
                    2 * self.generator.uniform(0, 1, self.problem.n_dims) * np.abs(stp)
            )
            ## Determine Three Random Indices of Solutions
            a, b, c = self.generator.choice(
                list(set(range(0, pop_size)) - {idx}), 3, replace=False
            )
            id_min_x = self.get_index_of_best_agent__(
                [self.population[a], self.population[b], self.population[c]]
            )
            ## Determine Xb and Xw for using in Runge Kutta method
            if cy.is_better(agent, self.population[id_min_x], self.problem.sense):
                xb, xw = agent.solution, self.population[id_min_x].solution
            else:
                xb, xw = self.population[id_min_x].solution, agent.solution
            ## Search Mechanism (SM) of RUN based on Runge Kutta Method
            SM = self.runge_kutta__(xb, xw, delta_x)
            local_best = cy.duplicate_agent(self.population.sort()[0])
            L = self.generator.choice(range(0, 2), self.problem.n_dims)
            xc = L * agent.solution + (1 - L) * self.population[a].solution  # Eq. 17.3
            xm = L * self.g_best.solution + (1 - L) * local_best.solution  # Eq. 17.4
            r = self.generator.choice([1, -1], self.problem.n_dims)  # An integer number
            g = 2 * self.generator.random()
            mu = 0.5 + 1 * self.generator.uniform(0, 1, self.problem.n_dims)
            ## Determine New Solution Based on Runge Kutta Method (Eq.18)
            if self.generator.random() < 0.5:
                x = xc + r * SF[idx] * g * xc + SF[idx] * SM + mu * (xm - xc)
            else:
                x = (
                        xm
                        + r * SF[idx] * g * xm
                        + SF[idx] * SM
                        + mu * (self.population[a].solution - self.population[b].solution)
                )
            x = cy.correct_solution(self.problem, x)
            tar_new = self.population.evaluate_solution(x)
            if cy.is_better(tar_new, agent, self.problem.sense):
                agent.update_solution(tar_new, x)
            ## Enhanced solution quality (ESQ)  (Eq. 19)
            if self.generator.random() < 0.5:
                w = self.uniform_random__(0, 2, self.problem.n_dims) * np.exp(
                    -5 * self.generator.random() * epoch / self.epoch
                )  # Eq.19-1
                r = np.floor(self.uniform_random__(-1, 2, 1))
                u = 2 * self.generator.random(self.problem.n_dims)
                a, b, c = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx}), 3, replace=False
                )
                x_ave = (
                                self.population[a].solution + self.population[b].solution + self.population[c].solution
                        ) / 3  # Eq.19-2
                beta = self.generator.random(self.problem.n_dims)
                x_new1 = beta * self.g_best.solution + (1 - beta) * x_ave  # Eq.19-3
                x_new2_temp1 = x_new1 + r * w * np.abs(
                    self.generator.normal(0, 1, self.problem.n_dims) + (x_new1 - x_ave)
                )
                x_new2_temp2 = (
                        x_new1
                        - x_ave
                        + r
                        * w
                        * np.abs(
                    self.generator.normal(0, 1, self.problem.n_dims)
                    + u * x_new1
                    - x_ave
                )
                )
                x_new2 = np.where(w < 1, x_new2_temp1, x_new2_temp2)
                pos_new2 = cy.correct_solution(self.problem, x_new2)
                tar_new2 = self.population.evaluate_solution(pos_new2)
                if cy.is_better(tar_new2, agent, self.problem.sense):
                    agent.update_solution(tar_new2, pos_new2)
                else:
                    if (
                            w[self.generator.integers(0, self.problem.n_dims)]
                            > self.generator.random()
                    ):
                        SM = self.runge_kutta__(
                            agent.solution, pos_new2, delta_x
                        )
                        x_new3 = (
                                pos_new2
                                - self.generator.random() * pos_new2
                                + SF[idx]
                                * (
                                        SM
                                        + (
                                                2
                                                * self.generator.random(self.problem.n_dims)
                                                * self.g_best.solution
                                                - pos_new2
                                        )
                                )
                        )  # Eq. 20
                        pos_new3 = cy.correct_solution(self.problem, x_new3)
                        tar_new3 = self.population.evaluate_solution(pos_new3)
                        if cy.is_better(tar_new3, agent, self.problem.sense):
                            agent.update_solution(tar_new3, pos_new3)
