cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 16:58, 08/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalGSKA(cy.Optimizer):
    """
    The original version of: Gaining Sharing Knowledge-based Algorithm (GSKA)

    Links:
        1. https://doi.org/10.1007/s13042-019-01053-x

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pb (float): [0.1, 0.5], percent of the best (p in the paper), default = 0.1
        + kf (float): [0.3, 0.8], knowledge factor that controls the total amount of gained and shared knowledge added from others to the current individual during generations, default = 0.5
        + kr (float): [0.5, 0.95], knowledge ratio, default = 0.9
        + kg (int): [3, 20], number of generations effect to D-dimension, default = 5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import GSKA    >>> import numpy as np
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
    >>> model = GSKA.OriginalGSKA(epoch=1000, pop_size=50, pb = 0.1, kf = 0.5, kr = 0.9, kg = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Mohamed, A.K., 2020. Gaining-sharing knowledge based algorithm for solving
    optimization problems: a novel nature-inspired algorithm. International Journal of Machine Learning and Cybernetics, 11(7), pp.1501-1529.
    """

    cdef public double kf
    cdef public int kg
    cdef public double kr
    cdef public double pb

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pb: float = 0.1,
            kf: float = 0.5,
            kr: float = 0.9,
            kg: int = 5,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100, n: pop_size, m: clusters
            pb (float): percent of the best   0.1%, 0.8%, 0.1% (p in the paper), default = 0.1
            kf (float): knowledge factor that controls the total amount of gained and shared knowledge added
                        from others to the current individual during generations, default = 0.5
            kr (float): knowledge ratio, default = 0.9
            kg (int): Number of generations effect to D-dimension, default = 5
        """
        super().__init__(parameters=["epoch", "pop_size", "pb", "kf", "kr", "kg"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pb = cy.validator(float, pb, (0, 1.0), "pb")
        self.kf = cy.validator(float, kf, (0, 1.0), "kf")
        self.kr = cy.validator(float, kr, (0, 1.0), "kr")
        self.kg = cy.validator(int, kg, [1, 1 + int(epoch / 2)], "kg")

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        dd = int(self.problem.n_dims * (1 - epoch / self.epoch) ** self.kg)
        pop_new = []
        for idx in range(0, pop_size):
            # If it is the best it chooses best+2, best+1
            if idx == 0:
                previ, nexti = idx + 2, idx + 1
            # If it is the worse it chooses worst-2, worst-1
            elif idx == pop_size - 1:
                previ, nexti = idx - 2, idx - 1
            # Other case it chooses i-1, i+1
            else:
                previ, nexti = idx - 1, idx + 1
            # The random individual is for all dimension values
            rand_idx = self.generator.choice(
                list(set(range(0, pop_size)) - {previ, idx, nexti})
            )
            pos_new = self.population[idx].solution.copy()

            for j in range(0, self.problem.n_dims):
                if j < dd:  # junior gaining and sharing
                    if self.generator.uniform() <= self.kr:
                        if cy.is_better(self.population[rand_idx], self.population[idx], self.problem.sense):
                            pos_new[j] = self.population[idx].solution[j] + self.kf * (
                                    self.population[previ].solution[j]
                                    - self.population[nexti].solution[j]
                                    + self.population[rand_idx].solution[j]
                                    - self.population[idx].solution[j]
                            )
                        else:
                            pos_new[j] = self.population[idx].solution[j] + self.kf * (
                                    self.population[previ].solution[j]
                                    - self.population[nexti].solution[j]
                                    + self.population[idx].solution[j]
                                    - self.population[rand_idx].solution[j]
                            )
                else:  # senior gaining and sharing
                    if self.generator.uniform() <= self.kr:
                        id1 = int(self.pb * pop_size)
                        id2 = int(id1 + pop_size * (1 - 2 * self.pb))
                        rand_best = self.generator.choice(
                            list(set(range(0, id1)) - {idx})
                        )
                        rand_worst = self.generator.choice(
                            list(set(range(id2, pop_size)) - {idx})
                        )
                        rand_mid = self.generator.choice(
                            list(set(range(id1, id2)) - {idx})
                        )
                        if cy.is_better(self.population[rand_mid], self.population[idx], self.problem.sense):
                            pos_new[j] = self.population[idx].solution[j] + self.kf * (
                                    self.population[rand_best].solution[j]
                                    - self.population[rand_worst].solution[j]
                                    + self.population[rand_mid].solution[j]
                                    - self.population[idx].solution[j]
                            )
                        else:
                            pos_new[j] = self.population[idx].solution[j] + self.kf * (
                                    self.population[rand_best].solution[j]
                                    - self.population[rand_worst].solution[j]
                                    + self.population[idx].solution[j]
                                    - self.population[rand_mid].solution[j]
                            )
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
