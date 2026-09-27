#!/usr/bin/env python
# Created by "Thieu" at 16:10, 08/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalArchOAAgent(cy.Agent):
    cdef public object den
    cdef public object vol
    cdef public object acc


cdef class OriginalArchOAPopulation(cy.Population):
    """Agents of :class:`OriginalArchOA`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        den = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)  # Density
        vol = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)  # Volume
        acc = self.problem.bounds.low + self.generator.uniform(
            self.problem.bounds.low, self.problem.bounds.up
        ) * (
            self.problem.bounds.up - self.problem.bounds.low
        )  # Acceleration
        return OriginalArchOAAgent(solution=solution, den=den, vol=vol, acc=acc)


cdef class OriginalArchOA(cy.Optimizer):
    """
    The original version of: Archimedes Optimization Algorithm (ArchOA)

    Links:
        1. https://doi.org/10.1007/s10489-020-01893-z

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (int): factor, default belongs to [1, 2]
        + c2 (int): factor, Default belongs to [2, 4, 6]
        + c3 (int): factor, Default belongs to [1, 2]
        + c4 (float): factor, Default belongs to [0.5, 1]
        + acc_max (float): acceleration max, Default 0.9
        + acc_min (float): acceleration min, Default 0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.physics_based import ArchOA    >>> import numpy as np
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
    >>> model = ArchOA.OriginalArchOA(epoch=1000, pop_size=50, c1 = 2, c2 = 5, c3 = 2, c4 = 0.5, acc_max = 0.9, acc_min = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Hashim, F.A., Hussain, K., Houssein, E.H., Mabrouk, M.S. and Al-Atabany, W., 2021. Archimedes optimization
    algorithm: a new metaheuristic algorithm for solving optimization problems. Applied Intelligence, 51(3), pp.1531-1551.
    """

    cdef public double acc_max
    cdef public double acc_min
    cdef public double c1
    cdef public double c2
    cdef public double c3
    cdef public double c4

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2,
        c2: float = 6,
        c3: float = 2,
        c4: float = 0.5,
        acc_max: float = 0.9,
        acc_min: float = 0.1,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c1 (float): factor, default belongs [1, 2]
            c2 (float): factor, Default belongs [2, 4, 6]
            c3 (float): factor, Default belongs [1, 2]
            c4 (float): factor, Default belongs [0.5, 1]
            acc_max (float): acceleration max, Default 0.9
            acc_min (float): acceleration min, Default 0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "c1", "c2", "c3", "c4", "acc_max", "acc_min"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalArchOAPopulation)
        self.c1 = cy.validator(float, c1, [1, 3], "c1")
        self.c2 = cy.validator(float, c2, [2, 6], "c2")
        self.c3 = cy.validator(float, c3, [1, 3], "c3")
        self.c4 = cy.validator(float, c4, (0, 1.0), "c4")
        self.acc_max = cy.validator(float, acc_max, (0.3, 1.0), "acc_max")
        self.acc_min = cy.validator(float, acc_min, (0, 0.3), "acc_min")

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Transfer operator Eq. 8
        tf = np.exp(epoch / self.epoch)
        ## Density decreasing factor Eq. 9
        ddf = np.exp(1.0 - epoch / self.epoch) - epoch / self.epoch
        list_acc = []
        ## Calculate new density, volume and acceleration
        for idx in range(0, pop_size):
            # Update density and volume of each object using Eq. 7
            new_den = self.population[idx].den + self.generator.uniform() * (
                self.g_best.den - self.population[idx].den
            )
            new_vol = self.population[idx].vol + self.generator.uniform() * (
                self.g_best.vol - self.population[idx].vol
            )
            # Exploration phase
            if tf <= 0.5:
                # Update acceleration using Eq. 10 and normalize acceleration using Eq. 12
                id_rand = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx})
                )
                new_acc = (
                    self.population[id_rand].den
                    + self.population[id_rand].vol * self.population[id_rand].acc
                ) / (new_den * new_vol)
            else:
                new_acc = (self.g_best.den + self.g_best.vol * self.g_best.acc) / (
                    new_den * new_vol
                )
            list_acc.append(new_acc)
            self.population[idx].den = new_den
            self.population[idx].vol = new_vol
        min_acc = np.min(list_acc)
        max_acc = np.max(list_acc)
        ## Normalize acceleration using Eq. 12
        for idx in range(0, pop_size):
            self.population[idx].acc = (
                self.acc_max
                * (list_acc[idx] - min_acc)
                / (max_acc - min_acc + self.EPSILON)
                + self.acc_min
            )
        pop_new = []
        for idx in range(0, pop_size):
            agent = self.population[idx].copy()
            if tf <= 0.5:  # update position using Eq. 13
                id_rand = self.generator.choice(
                    list(set(range(0, pop_size)) - {idx})
                )
                pos_new = self.population[
                    idx
                ].solution + self.c1 * self.generator.uniform() * self.population[
                    idx
                ].acc * ddf * (
                    self.population[id_rand].solution - self.population[idx].solution
                )
            else:
                p = 2 * self.generator.random() - self.c4
                f = 1 if p <= 0.5 else -1
                t = self.c3 * tf
                pos_new = (
                    self.g_best.solution
                    + f
                    * self.c2
                    * self.generator.random()
                    * self.population[idx].acc
                    * ddf
                    * (t * self.g_best.solution - self.population[idx].solution)
                )
            agent.solution = self.population.correct_solution(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                # the classic code evaluates pos_new, not agent.solution (MEALPY behaviour, kept)
                agent.update_solution(self.population.evaluate_solution(pos_new), agent.solution)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
