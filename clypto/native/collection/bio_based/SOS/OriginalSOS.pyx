cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 14:20, 15/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalSOS(cy.Optimizer):
    """
    The original version: Symbiotic Organisms Search (SOS)

    Links:
        1. https://doi.org/10.1016/j.compstruc.2014.03.007

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.bio_based import SOS    >>> import numpy as np
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
    >>> model = SOS.OriginalSOS(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Cheng, M. Y., & Prayogo, D. (2014). Symbiotic organisms search: a new metaheuristic
    optimization algorithm. Computers & Structures, 139, 98-112.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx, agent in enumerate(self.population.toarray()):
            ## Mutualism Phase
            jdx = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            mutual_vector = (agent.solution + self.population[jdx].solution) / 2
            bf1, bf2 = self.generator.integers(1, 3, 2)
            xi_new = agent.solution + self.generator.random() * (
                    self.g_best.solution - bf1 * mutual_vector
            )
            xj_new = self.population[jdx].solution + self.generator.random() * (
                    self.g_best.solution - bf2 * mutual_vector
            )
            xi_new = cy.correct_solution(self.problem, xi_new)
            xj_new = cy.correct_solution(self.problem, xj_new)
            xi_target = self.population.evaluate_solution(xi_new)
            xj_target = self.population.evaluate_solution(xj_new)
            if cy.is_better(xi_target, agent, self.problem.sense):
                agent.update_solution(xi_target, xi_new)
            if cy.is_better(xj_target, self.population[jdx], self.problem.sense):
                self.population[jdx].update_solution(xj_target, xj_new)
            ## Commensalism phase
            jdx = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            xi_new = agent.solution + self.generator.uniform(-1, 1) * (
                    self.g_best.solution - self.population[jdx].solution
            )
            xi_new = cy.correct_solution(self.problem, xi_new)
            xi_target = self.population.evaluate_solution(xi_new)
            if cy.is_better(xi_target, agent, self.problem.sense):
                agent.update_solution(xi_target, xi_new)
            ## Parasitism phase
            jdx = self.generator.choice(list(set(range(0, pop_size)) - {idx}))
            temp_idx = self.generator.integers(0, self.problem.n_dims)
            xi_new = self.population[jdx].solution.copy()
            xi_new[temp_idx] = self.problem.generate_solution()[temp_idx]
            xi_new = cy.correct_solution(self.problem, xi_new)
            xi_target = self.population.evaluate_solution(xi_new)
            if cy.is_better(xi_target, agent, self.problem.sense):
                agent.update_solution(xi_target, xi_new)
