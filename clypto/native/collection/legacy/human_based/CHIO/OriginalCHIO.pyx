#!/usr/bin/env python
# Created by "Thieu" at 19:24, 09/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalCHIO(cy.Optimizer):
    """
    The original version of: Coronavirus Herd Immunity Optimization (CHIO)

    Links:
        1. https://link.springer.com/article/10.1007/s00521-020-05296-6

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + brr (float): [0.05, 0.2], Basic reproduction rate, default=0.15
        + max_age (int): [5, 20], Maximum infected cases age, default=10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import CHIO    >>> import numpy as np
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
    >>> model = CHIO.OriginalCHIO(epoch=1000, pop_size=50, brr = 0.15, max_age = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Al-Betar, M.A., Alyasseri, Z.A.A., Awadallah, M.A. et al. Coronavirus herd immunity optimizer (CHIO).
    Neural Comput & Applic 33, 5011–5042 (2021). https://doi.org/10.1007/s00521-020-05296-6
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            brr: float = 0.15,
            max_age: int = 10,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            brr (float): Basic reproduction rate, default=0.15
            max_age (int): Maximum infected cases age, default=10
        """
        super().__init__(parameters=["epoch", "pop_size", "brr", "max_age"], **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.brr = cy.validator(float, brr, (0, 1.0), "brr")
        self.max_age = cy.validator(int, max_age, [1, 1 + int(epoch / 5)], "max_age")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.immunity_type_list = self.generator.integers(
            0, 3, pop_size
        )  # Randint [0, 1, 2]
        self.age_list = np.zeros(pop_size)  # Control the age of each position
        self.finished = False

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        is_corona_list = [
                             False,
                         ] * pop_size
        for i in range(0, pop_size):
            pos_new = self.population[i].solution.copy()
            for j in range(0, self.problem.n_dims):
                rand = self.generator.uniform()
                if rand < (1.0 / 3) * self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 1
                    )  # Infected list
                    if idx_candidates[0].size == 0:
                        self.finished = True
                        # print("Epoch: {}, i: {}, immunity_list: {}".format(epoch, i, self.immunity_type_list))
                        break
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.population[i].solution[j] + self.generator.uniform() * (
                            self.population[i].solution[j] - self.population[idx_selected].solution[j]
                    )
                    is_corona_list[i] = True
                elif (1.0 / 3) * self.brr <= rand < (2.0 / 3) * self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 0
                    )  # Susceptible list
                    idx_selected = self.generator.choice(idx_candidates[0])
                    pos_new[j] = self.population[i].solution[j] + self.generator.uniform() * (
                            self.population[i].solution[j] - self.population[idx_selected].solution[j]
                    )
                elif (2.0 / 3) * self.brr <= rand < self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 2
                    )  # Immunity list
                    fit_list = np.array(
                        [self.population[item].fitness for item in idx_candidates[0]]
                    )
                    idx_selected = idx_candidates[0][
                        np.argmin(fit_list)
                    ]  # Found the index of best fitness
                    pos_new[j] = self.population[i].solution[j] + self.generator.uniform() * (
                            self.population[i].solution[j] - self.population[idx_selected].solution[j]
                    )
            if self.finished:
                break
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        pop_new = self.population.evaluate(pop_new, self.mode)
        if len(pop_new) != pop_size:
            pop_child = self.population.generate(pop_size - len(pop_new))
            pop_new = pop_new + pop_child
        for idx in range(0, pop_size):
            # Step 4: Update herd immunity population
            if cy.is_better(pop_new[idx], self.population[idx], self.problem.sense):
                self.population[idx] = pop_new[idx].copy()
            else:
                self.age_list[idx] += 1
            ## Calculate immunity mean of population
            fit_list = np.array([agent.fitness for agent in self.population])
            delta_fx = np.mean(fit_list)
            if (
                    cy.better_fitness(pop_new[idx].fitness, delta_fx, self.problem.sense)
                    and self.immunity_type_list[idx] == 0
                    and is_corona_list[idx]
            ):
                self.immunity_type_list[idx] = 1
                self.age_list[idx] = 1
            if cy.better_fitness(delta_fx, pop_new[idx].fitness, self.problem.sense) and (self.immunity_type_list[idx] == 1):
                self.immunity_type_list[idx] = 2
                self.age_list[idx] = 0
            # Step 5: Fatality condition
            if (self.age_list[idx] >= self.max_age) and (
                    self.immunity_type_list[idx] == 1
            ):
                self.population[idx] = self.population.generate_agent()
                self.immunity_type_list[idx] = 0
                self.age_list[idx] = 0
