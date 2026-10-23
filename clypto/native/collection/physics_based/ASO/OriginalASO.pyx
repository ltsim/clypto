#!/usr/bin/env python
# Created by "Thieu" at 07:03, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalASOAgent(cy.Agent):
    cdef public object velocity
    cdef public object mass

    def __init__(self, solution=None, objectives=None, weights=None, velocity=None, mass=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.velocity = velocity
        self.mass = mass

    cdef cy.Agent clone(self):
        cdef OriginalASOAgent new = <OriginalASOAgent>cy.Agent.clone(self)
        new.velocity = self.velocity
        new.mass = self.mass
        return new


cdef class OriginalASOPopulation(cy.Population):
    """Agents of :class:`OriginalASO`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        mass = 0.0
        return OriginalASOAgent(solution=solution, velocity=velocity, mass=mass)


cdef class OriginalASO(cy.Optimizer):
    """
    The original version of: Atom Search Optimization (ASO)

    Links:
        1. https://doi.org/10.1016/j.knosys.2018.08.030
        2. https://www.mathworks.com/matlabcentral/fileexchange/67011-atom-search-optimization-aso-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (int): Depth weight, default = 10, depend on the problem
        + beta (float): Multiplier weight, default = 0.2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import ASO    >>> import numpy as np
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
    >>> model = ASO.OriginalASO(epoch=1000, pop_size=50, alpha = 50, beta = 0.2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Zhao, W., Wang, L. and Zhang, Z., 2019. Atom search optimization and its application to solve a
    hydrogeologic parameter estimation problem. Knowledge-Based Systems, 163, pp.283-304.
    """

    cdef public int alpha
    cdef public double beta

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        alpha: int = 10,
        beta: float = 0.2,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): [2, 20], Depth weight, default = 10
            beta (float): [0.1, 1.0], Multiplier weight, default = 0.2
        """
        super().__init__(parameters=["epoch", "pop_size", "alpha", "beta"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalASOPopulation)
        self.alpha = cy.validator(int, alpha, [1, 100], "alpha")
        self.beta = cy.validator(float, beta, (0, 1.0), "beta")

    def update_mass__(self, population):
        pop_size = self.population.size()
        list_fit = np.array([agent.fitness for agent in population])
        list_fit = np.exp(
            -(list_fit - np.max(list_fit))
            / (np.max(list_fit) - np.min(list_fit) + self.EPSILON)
        )
        list_fit = list_fit / np.sum(list_fit)
        for idx in range(0, pop_size):
            population[idx].mass = list_fit[idx]
        return population

    def find_LJ_potential__(self, iteration, average_dist, radius):
        c = (1 - iteration / self.epoch) ** 3
        # g0 = 1.1, u = 2.4
        rsmin = 1.1 + 0.1 * np.sin(iteration / self.epoch * np.pi / 2)
        rsmax = 1.24
        if radius / average_dist < rsmin:
            rs = rsmin
        else:
            if radius / average_dist > rsmax:
                rs = rsmax
            else:
                rs = radius / average_dist
        potential = c * (12 * (-rs) ** (-13) - 6 * (-rs) ** (-7))
        return potential

    def acceleration__(self, population, g_best, iteration):
        pop_size = self.population.size()
        eps = 2.0 ** (-52)
        pop = self.update_mass__(population)
        G = np.exp(-20.0 * iteration / self.epoch)
        k_best = (
            int(pop_size - (pop_size - 2) * (iteration / self.epoch) ** 0.5)
            + 1
        )
        if self.problem.sense == "min":
            k_best_pop = sorted(pop, key=lambda agent: agent.mass, reverse=True)[
                :k_best
            ].copy()
        else:
            k_best_pop = sorted(pop, key=lambda agent: agent.mass)[:k_best].copy()
        mk_average = np.mean([agent.solution for agent in k_best_pop])
        acc_list = np.zeros((pop_size, self.problem.n_dims))
        for idx in range(0, pop_size):
            dist_average = np.linalg.norm(pop[idx].solution - mk_average)
            temp = np.zeros((self.problem.n_dims))
            for atom in k_best_pop:
                # calculate LJ-potential
                radius = np.linalg.norm(pop[idx].solution - atom.solution)
                potential = self.find_LJ_potential__(iteration, dist_average, radius)
                temp += (
                    potential
                    * self.generator.uniform(0, 1, self.problem.n_dims)
                    * ((atom.solution - pop[idx].solution) / (radius + eps))
                )
            temp = self.alpha * temp + self.beta * (g_best.solution - pop[idx].solution)
            # calculate acceleration
            acc = G * temp / pop[idx].mass
            acc_list[idx] = acc
        return acc_list

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Calculate acceleration.
        atom_acc_list = self.acceleration__(self.population, self.g_best, iteration=epoch)
        # Update velocity based on random dimensions and position of global best
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            child = cy.duplicate_agent(agent)
            velocity = (
                self.generator.random(self.problem.n_dims) * agent.velocity
                + atom_acc_list[idx]
            )
            x = agent.solution + velocity
            # Relocate atom out of range
            x = cy.reset_solution(self.problem, self.generator, x)
            child.solution = x
            n_population.append(child)
            if self.mode == "sequential":
                child.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(child, self.population[idx], self.problem.sense)
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            self.population = self.population.greedy(n_population)
        current_best = cy.duplicate_agent(cy.sort_agents(n_population, self.problem.sense)[0])
        if cy.is_better(self.g_best, current_best, self.problem.sense):
            self.population[self.generator.integers(0, pop_size)] = cy.duplicate_agent(self.g_best)
