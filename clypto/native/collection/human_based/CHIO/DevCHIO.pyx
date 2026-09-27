#!/usr/bin/env python
# Created by "Thieu" at 19:24, 09/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy

from clypto.native.collection.human_based.CHIO.OriginalCHIO cimport OriginalCHIO


cdef class DevCHIO(OriginalCHIO):
    """
    The developed version of: Coronavirus Herd Immunity Optimization (CHIO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + brr (float): [0.05, 0.2], Basic reproduction rate, default=0.15
        + max_age (int): [5, 20], Maximum infected cases age, default=10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import CHIO    >>> import numpy as np
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
    >>> model = CHIO.DevCHIO(epoch=1000, pop_size=50, brr = 0.15, max_age = 10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
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
        super().__init__(epoch, pop_size, brr, max_age, **kwargs)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        is_corona_list = [
                             False,
                         ] * pop_size
        for i in range(0, pop_size):
            x = self.population[i].solution.copy()
            for j in range(0, self.problem.n_dims):
                rand = self.generator.uniform()
                if rand < (1.0 / 3) * self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 1
                    )  # Infected list
                    if idx_candidates[0].size == 0:
                        rand_choice = self.generator.choice(
                            range(0, pop_size),
                            int(0.33 * pop_size),
                            replace=False,
                        )
                        self.immunity_type_list[rand_choice] = 1
                        idx_candidates = np.where(self.immunity_type_list == 1)
                    idx_selected = self.generator.choice(idx_candidates[0])
                    x[j] = self.population[i].solution[j] + self.generator.uniform() * (
                            self.population[i].solution[j] - self.population[idx_selected].solution[j]
                    )
                    is_corona_list[i] = True
                elif (1.0 / 3) * self.brr <= rand < (2.0 / 3) * self.brr:
                    idx_candidates = np.where(
                        self.immunity_type_list == 0
                    )  # Susceptible list
                    if idx_candidates[0].size == 0:
                        rand_choice = self.generator.choice(
                            range(0, pop_size),
                            int(0.33 * pop_size),
                            replace=False,
                        )
                        self.immunity_type_list[rand_choice] = 0
                        idx_candidates = np.where(self.immunity_type_list == 0)
                    idx_selected = self.generator.choice(idx_candidates[0])
                    x[j] = self.population[i].solution[j] + self.generator.uniform() * (
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
                    x[j] = self.population[i].solution[j] + self.generator.uniform() * (
                            self.population[i].solution[j] - self.population[idx_selected].solution[j]
                    )
            if self.finished:
                break
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
        n_population = self.population.evaluate(n_population, self.mode)

        for idx, agent in enumerate(self.population.toarray()):
            # Step 4: Update herd immunity population
            if cy.is_better(n_population[idx], agent, self.problem.sense):
                self.population[idx] = cy.duplicate_agent(n_population[idx])
            else:
                self.age_list[idx] += 1
            ## Calculate immunity mean of population
            fit_list = np.array([child.fitness for child in self.population])
            delta_fx = np.mean(fit_list)
            if (
                    cy.better_fitness(n_population[idx].fitness, delta_fx, self.problem.sense)
                    and (self.immunity_type_list[idx] == 0)
                    and is_corona_list[idx]
            ):
                self.immunity_type_list[idx] = 1
                self.age_list[idx] = 1
            if cy.better_fitness(delta_fx, n_population[idx].fitness, self.problem.sense) and (self.immunity_type_list[idx] == 1):
                self.immunity_type_list[idx] = 2
                self.age_list[idx] = 0
            # Step 5: Fatality condition
            if (self.age_list[idx] >= self.max_age) and (
                    self.immunity_type_list[idx] == 1
            ):
                self.population[idx] = self.population.generate_agent()
                self.immunity_type_list[idx] = 0
                self.age_list[idx] = 0
