cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 23:41, 15/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalCDDO(cy.Optimizer):
    """
    The original version of: Child Drawing Development Optimization (CCDO)

    Notes:
        + This source code was converted from the original Matlab implementation in the paper into Python.
        The Matlab code itself has many issues, for example, parameters are defined but never used.
        Several variables are declared, such as p1, p2, p3. Parameters like child skill rate and child level
        rate are initialized as hyperparameters at the beginning, but inside the loop they are randomly generated,
        which is inconsistent with the paper.

        + Moreover, the biggest flaw of this algorithm lies in the if–else condition during the update process.
        There is a high chance that neither condition will be executed, because the golden ratio is not necessarily
        within the interval [1.5, 2], as it is computed based on a random position. In addition, when comparing
        the position with a random integer T (hand pressure), it is unclear why this is done. It is highly likely
        that the algorithm will only execute that single condition.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import CDDO    >>> import numpy as np
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
    >>> model = CDDO.OriginalCDDO(epoch=1000, pop_size=50, pattern_size=10, creativity_rate=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdulhameed, S., Rashid, T.A. Child Drawing Development Optimization Algorithm Based on
    Child’s Cognitive Development. Arab J Sci Eng 47, 1337–1351 (2022). https://doi.org/10.1007/s13369-021-05928-6
    """

    cdef public double creativity_rate
    cdef public int pattern_size

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            pattern_size=10,
            creativity_rate=0.1,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pattern_size (int): size of the pattern matrix, default = 10
            creativity_rate (float): creativity rate, default = 0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "pattern_size", "creativity_rate"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pattern_size = cy.validator(int, pattern_size, [1, 1000], "pattern_size")
        self.creativity_rate = cy.validator(float, creativity_rate, [0.0, 1.0], "creativity_rate")

    def before_main_loop(self):
        pop_size = self.population.size()
        self.LR = self.generator.uniform(0.1, 1.0)  # Child level rate
        self.SR = self.generator.uniform(0.1, 1.0)  # Child Skill Rate
        self.pop_local = self.population.copy()
        # Golden ratio
        self.list_gr = []
        for idx in range(pop_size):
            p1 = self.generator.integers(0, self.problem.n_dims)
            p2 = self.generator.integers(0, self.problem.n_dims)
            if self.population[idx].solution[p1] == 0:
                self.list_gr.append(self.population[idx].solution[p2])
            else:
                self.list_gr.append(
                    self.population[idx].solution[p1]
                    + self.population[idx].solution[p2] / self.population[idx].solution[p1]
                )

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Pattern matrix
        ranked = self.population.sort()
        pattern = [agent.copy() for agent in ranked[:self.pattern_size]]
        for idx in range(0, pop_size):
            hand_pressure = self.generator.integers(
                self.problem.bounds.low[0], self.problem.bounds.up[0] + 1
            )
            pp = self.generator.integers(0, self.problem.n_dims)
            pos_new = self.population[idx].solution.copy()
            if self.population[idx].solution[pp] <= hand_pressure:
                # Update the drawings
                pos_new = (
                        self.list_gr[idx]
                        + self.SR
                        * self.generator.random(self.problem.n_dims)
                        * (self.pop_local[idx].solution - self.population[idx].solution)
                        + self.LR
                        * self.generator.random(self.problem.n_dims)
                        * (self.g_best.solution - self.population[idx].solution)
                )
                self.LR = self.generator.integers(6, 11) / 10
                self.SR = self.generator.integers(6, 11) / 10
            elif 1.5 < self.list_gr[idx] < 2:
                # Consider the learnt patterns
                pos_new = (
                        pattern[self.generator.integers(0, self.pattern_size)].solution
                        - self.creativity_rate * self.pop_local[idx].solution
                )
                self.LR = self.generator.integers(0, 6) / 10
                self.SR = self.generator.integers(0, 6) / 10
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            self.population[idx] = agent
            if self.mode not in self.AVAILABLE_MODES:
                self.population[idx].evaluate(self.problem)
        if self.mode in self.AVAILABLE_MODES:
            self.population = self.population.evaluate(self.population, self.mode)
        # Update the local information
        self.pop_local = cy.greedy_agents(self.pop_local, self.population, self.problem.sense)
