#!/usr/bin/env python
# Created by "Thieu" at 11:40, 20/12/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalBWO(cy.Optimizer):
    """
    The original version of: Black Widow Optimization (BWO)

    Links:
        1. https://doi.org/10.1016/j.engappai.2019.103249

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pp (float): [0, 1], procreating rate, default = 0.6
        + cr (float): [0, 1], cannibalism rate, default = 0.44
        + pm (float): [0, 1], mutation rate, default = 0.4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import BWO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function,
    >>> }
    >>>
    >>> model = BWO.OriginalBWO(epoch=1000, pop_size=50, pp=0.6, cr=0.44, pm=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Hayyolalam, V. and Pourhaji Kazem, A.A., 2020. Black widow optimization algorithm: A novel meta-heuristic
    approach for solving engineering optimization problems. Engineering Applications of Artificial Intelligence, 87, 103249.
    """

    cdef public double cr
    cdef public double pm
    cdef public double pp

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pp: float = 0.6,
            cr: float = 0.44,
            pm: float = 0.4,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pp (float): procreating rate, default = 0.6
            cr (float): cannibalism rate, default = 0.44
            pm (float): mutation rate, default = 0.4
        """
        super().__init__(parameters=["epoch", "pop_size", "pp", "cr", "pm"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pp = cy.validator(float, pp, (0.0, 1.0), "pp")
        self.cr = cy.validator(float, cr, (0.0, 1.0), "cr")
        self.pm = cy.validator(float, pm, (0.0, 1.0), "pm")

    def initialize_variables(self):
        pop_size = self.population.size()
        self.n_parents = max(2, int(self.pp * pop_size))
        if self.n_parents > pop_size:
            self.n_parents = pop_size
        self.n_mutate = max(0, int(self.pm * pop_size))

    def _procreate(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """
        Create two offspring from a pair of parents using blend crossover on Nvar/2 indices.
        """
        n_dims = self.problem.n_dims
        n_cross = max(1, n_dims // 2)
        idxs = self.generator.choice(n_dims, n_cross, replace=False)
        alpha = self.generator.random(len(idxs))
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[idxs] = alpha * parent1[idxs] + (1 - alpha) * parent2[idxs]
        child2[idxs] = alpha * parent2[idxs] + (1 - alpha) * parent1[idxs]
        return cy.correct_solution(self.problem, child1), cy.correct_solution(self.problem, child2)

    def _mutate(self, position: np.ndarray) -> np.ndarray:
        """
        Mutate one randomly selected position in the solution vector.
        """
        if self.problem.n_dims < 1:
            return position
        pos_new = position.copy()
        idx = self.generator.integers(0, self.problem.n_dims)
        pos_new[idx] = self.generator.uniform(
            self.problem.bounds.low[idx], self.problem.bounds.up[idx]
        )
        return cy.correct_solution(self.problem, pos_new)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_sorted = self.population.sort()
        pop1 = [cy.duplicate_agent(agent) for agent in pop_sorted[: self.n_parents]]

        pop2 = []
        for _ in range(self.n_parents):
            parent_idx = self.generator.choice(len(pop1), 2, replace=False)
            parent1, parent2 = pop1[parent_idx[0]], pop1[parent_idx[1]]
            female = cy.duplicate_agent(cy.get_better_agent(parent1, parent2, self.problem.sense))
            child1_pos, child2_pos = self._procreate(parent1.solution, parent2.solution)
            child1 = self.population.create_agent(child1_pos)
            child2 = self.population.create_agent(child2_pos)
            children = [child1, child2]
            if self.mode != "sequential":
                self.population.evaluate(children, self.mode)
            else:
                for child in children:
                    child.evaluate(self.problem)
            n_keep = self.generator.binomial(len(children), 1 - self.cr)
            if n_keep < 1:
                n_keep = 1
            children = cy.sort_agents(children, self.problem.sense)
            pop2.append(female)
            pop2.extend(children[:n_keep])

        pop3 = []
        if self.n_mutate > 0:
            for _ in range(self.n_mutate):
                parent = pop1[self.generator.integers(0, len(pop1))]
                x = self._mutate(parent.solution)
                pop3.append(self.population.create_agent(x))
            if self.mode != "sequential":
                self.population.evaluate(pop3, self.mode)
            else:
                for agent in pop3:
                    agent.evaluate(self.problem)

        pop_new = pop2 + pop3
        if len(pop_new) < pop_size:
            needed = pop_size - len(pop_new)
            pop_new.extend([cy.duplicate_agent(agent) for agent in pop_sorted[:needed]])
        self.population = self.population.spawn(cy.sort_agents(pop_new, self.problem.sense)[:pop_size])
