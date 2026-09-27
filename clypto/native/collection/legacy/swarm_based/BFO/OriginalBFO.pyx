#!/usr/bin/env python
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalBFOAgent(cy.Agent):
    cdef public object cost
    cdef public object interaction
    cdef public object nutrients


cdef class OriginalBFOPopulation(cy.Population):
    """Agents of :class:`OriginalBFO`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        cost = 0.0
        interaction = 0.0
        nutrients = 0.0
        return OriginalBFOAgent(
            solution=solution, cost=cost, interaction=interaction, nutrients=nutrients
        )


cdef class OriginalBFO(cy.Optimizer):
    """
    The original version of: Bacterial Foraging Optimization (BFO)

    Notes:
        + Ned and Nre parameters are replaced by epoch (generation)
        + The Nc parameter will also decrease to reduce the computation time.
        + Cost in this version equal to Fitness value in the paper.
        + https://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ci (float): [0.01, 0.3], step size, default=0.01
        + Ped (float): [0.1, 0.5], probability of elimination, default=0.25
        + Ned (int): elim_disp_steps (Removed), Ned=5,
        + Nre (int): reproduction_steps (Removed), Nre=50,
        + Nc (int): [3, 10], chem_steps (Reduce), Nc = Original Nc/2, default = 5
        + Ns (int): [2, 10], swim length, default=4
        + d_attract (float): coefficient to calculate attract force, default = 0.1
        + w_attract (float): coefficient to calculate attract force, default = 0.2
        + h_repels (float): coefficient to calculate repel force, default = 0.1
        + w_repels (float): coefficient to calculate repel force, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BFO    >>> import numpy as np
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
    >>> model = BFO.OriginalBFO(epoch=1000, pop_size=50, Ci = 0.01, Ped = 0.25, Nc = 5, Ns = 4, d_attract=0.1, w_attract=0.2, h_repels=0.1, w_repels=10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Passino, K.M., 2002. Biomimicry of bacterial foraging for distributed optimization and control.
    IEEE control systems magazine, 22(3), pp.52-67.
    """

    cdef public int Ci
    cdef public int Nc
    cdef public int Ns
    cdef public double Ped
    cdef public double d_attract
    cdef public double h_repels
    cdef public double w_attract
    cdef public double w_repels

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        Ci: float = 0.01,
        Ped: float = 0.25,
        Nc: int = 5,
        Ns: int = 4,
        d_attract: float = 0.1,
        w_attract: float = 0.2,
        h_repels: float = 0.1,
        w_repels: float = 10,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ci (float): step size, default=0.01
            Ped (float): p_eliminate, default=0.25
            Ned (int): elim_disp_steps (Removed)         Ned=5,
            Nre (int): reproduction_steps (Removed)      Nre=50,
            Nc (int): chem_steps (Reduce)                Nc = Original Nc/2, default = 5
            Ns (int): swim_length, default=4
            d_attract (float): coefficient to calculate attract force, default = 0.1
            w_attract (float): coefficient to calculate attract force, default = 0.2
            h_repels (float): coefficient to calculate repel force, default = 0.1
            w_repels (float): coefficient to calculate repel force, default = 10
        """
        super().__init__(parameters=[ "epoch", "pop_size", "Ci", "Ped", "Nc", "Ns", "d_attract", "w_attract", "h_repels", "w_repels", ], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalBFOPopulation)
        self.step_size = self.Ci = cy.validator(int, Ci, (0, 5.0), "Ci")
        self.p_eliminate = self.Ped = cy.validator(float, Ped, (0, 1.0), "Ped")
        self.chem_steps = self.Nc = cy.validator(int, Nc, [2, 100], "Nc")
        self.swim_length = self.Ns = cy.validator(int, Ns, [2, 100], "Ns")
        self.d_attract = cy.validator(float, d_attract, (0, 1.0), "d_attract")
        self.w_attract = cy.validator(float, w_attract, (0, 1.0), "w_attract")
        self.h_repels = cy.validator(float, h_repels, (0, 1.0), "h_repels")
        self.w_repels = cy.validator(float, w_repels, (2.0, 20.0), "w_repels")
        self.half_pop_size = int(self.population.size() / 2)

    def compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem.n_dims * ((cell.solution - other.solution) ** 2).mean(
                axis=None
            )
            sum_inter += d * np.exp(w * diff)
        return sum_inter

    def attract_repel__(self, idx, cells):
        attract = self.compute_cell_interaction__(
            cells[idx], cells, -self.d_attract, -self.w_attract
        )
        repel = self.compute_cell_interaction__(
            cells[idx], cells, self.h_repels, -self.w_repels
        )
        return attract + repel

    def evaluate__(self, idx, cells):
        cells[idx].interaction = self.attract_repel__(idx, cells)
        cells[idx].cost = cells[idx].fitness + cells[idx].interaction
        return cells

    def tumble_cell__(self, cell, step_size):
        delta_i = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
        vector = cell.solution + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for j in range(0, self.chem_steps):
            for idx in range(0, pop_size):
                sum_nutrients = 0.0
                self.population = self.evaluate__(idx, self.population)
                sum_nutrients += self.population[idx].cost

                for m in range(0, self.swim_length):
                    delta_i = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
                    unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                    pos_new = self.population[idx].solution + self.step_size * unit_vector
                    pos_new = self.population.correct_solution(pos_new)
                    agent = self.population.generate_agent(pos_new)
                    if cy.is_better(agent, self.population[idx], self.problem.sense):
                        self.population[idx] = agent
                        break
                    sum_nutrients += self.population[idx].cost
                self.population[idx].nutrients = sum_nutrients
            cells = sorted(self.population, key=lambda cell: cell.nutrients)
            self.population = (
                cells[0 : self.half_pop_size].copy()
                + cells[0 : pop_size - self.half_pop_size].copy()
            )
            for idc in range(pop_size):
                if self.generator.random() < self.p_eliminate:
                    self.population[idc] = self.population.generate_agent()
