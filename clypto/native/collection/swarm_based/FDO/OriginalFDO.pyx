#!/usr/bin/env python
# Created by "Thieu" at 10:01, 16/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalFDO(cy.Optimizer):
    """
    The original version of: Fitness Dependent Optimizer (FDO)

    Notes:
        + https://doi.org/10.1109/ACCESS.2019.2907012
        + Inspired by the bee swarming reproductive process, this algorithm optimizes solutions based on their fitness values.
        + This algorithm mainly relies on Lévy flight techniques. Thanks to this method of generating random numbers
        according to the Lévy distribution, it is able to converge. However, in the design of the fitness weight
        condition, it is almost impossible for an update to occur when the fitness weight equals 1. This is the main drawback.

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import FDO    >>> import numpy as np
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
    >>> model = FDO.OriginalFDO(epoch=1000, pop_size=50, weight_factor=0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Abdullah, J. M., & Ahmed, T. (2019).
    Fitness dependent optimizer: inspired by the bee swarming reproductive process. IEEe Access, 7, 43473-43486.
    """

    cdef public double weight_factor

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            weight_factor=0.1,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            weight_factor (float): factor to adjust the fitness weight calculation, default = 0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "weight_factor"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.weight_factor = cy.validator(float, weight_factor, [0.0, 1.0], "weight_factor")

    def before_main_loop(self):
        pop_size = self.population.size()
        self.pop_pace = [
                            0,
                        ] * pop_size

    def get_fit_weight(self, best_fit, current_fit, weight_factor=0.1):
        """
        Calculate the fitness weight based on the best and current fitness values.

        Args:
            best_fit (float): The best fitness value found so far.
            current_fit (float): The current fitness value of the agent.
            weight_factor (float): A factor to adjust the weight calculation, default is 0.1.

        Returns:
            float: The fitness weight.
        """
        if best_fit == 0:
            return 0
        else:
            if self.problem.sense == "min":
                if best_fit < (0.05 * current_fit):
                    return 0.2
                else:
                    return best_fit / current_fit - weight_factor
            else:
                if best_fit > (0.05 * current_fit):
                    return 0.2
                else:
                    return weight_factor - best_fit / current_fit

    def get_into_levy_bound(self, pos_new):
        """
        Ensure the new position is within the levy bounds.

        Args:
            pos_new (np.ndarray): The new position to be checked.

        Returns:
            np.ndarray: The position clipped to the problem bounds.
        """
        levy = cy.levy_flight(self.generator, beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
        levy_up = self.problem.bounds.up * np.abs(levy)
        levy_lb = self.problem.bounds.low * np.abs(levy)
        pos_new = np.select(
            [pos_new > self.problem.bounds.up, pos_new < self.problem.bounds.low],
            [levy_up, levy_lb],
            default=pos_new,
        )
        return pos_new

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Update positions for each thief
        for idx, agent in enumerate(self.population.toarray()):
            fw = self.get_fit_weight(
                self.g_best.fitness,
                agent.fitness,
                self.weight_factor,
            )
            dist = self.g_best.solution - agent.solution
            levy = cy.levy_flight(self.generator, beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
            if fw == 1:
                pace = agent.solution * levy
            elif fw == 0:
                pace = dist * levy
            else:
                pace = dist * fw * np.sign(levy)
            self.pop_pace[idx] = pace
            x = agent.solution + pace
            x = self.get_into_levy_bound(x)
            x = cy.correct_solution(self.problem, x)
            child = self.population.generate_agent(x)
            # Check if new position is better
            if cy.is_better(child, agent, self.problem.sense):
                self.population[idx] = child
            else:
                # Alternative update strategy
                dist = self.g_best.solution - x
                x = x + (dist * fw) + self.pop_pace[idx]
                x = self.get_into_levy_bound(x)
                x = cy.correct_solution(self.problem, x)
                child = self.population.generate_agent(x)
                if cy.is_better(child, self.population[idx], self.problem.sense):
                    self.population[idx] = child
                else:
                    # Third update strategy
                    levy = cy.levy_flight(self.generator, beta=1.5, multiplier=0.01, size=self.problem.n_dims, case=-1)
                    x = self.population[idx].solution + self.population[idx].solution * levy
                    x = self.get_into_levy_bound(x)
                    x = cy.correct_solution(self.problem, x)
                    child = self.population.generate_agent(x)
                    if cy.is_better(child, self.population[idx], self.problem.sense):
                        self.population[idx] = child
