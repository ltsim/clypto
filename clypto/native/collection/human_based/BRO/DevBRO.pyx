#!/usr/bin/env python
# Created by "Thieu" at 09:17, 09/11/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from scipy.spatial.distance import cdist
cimport clypto.core as cy



cdef class DevBROAgent(cy.Agent):
    cdef public object damage

    def __init__(self, solution=None, objectives=None, weights=None, damage=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.damage = damage

    cdef cy.Agent clone(self):
        cdef DevBROAgent new = <DevBROAgent>cy.Agent.clone(self)
        new.damage = self.damage
        return new


cdef class DevBROPopulation(cy.Population):
    """Agents of :class:`DevBRO`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        damage = 0
        return DevBROAgent(solution=solution, damage=damage)


cdef class DevBRO(cy.Optimizer):
    """
    The developed version: Battle Royale Optimization (BRO)

    Notes:
        + The flow of algorithm is changed. Thrid loop is removed

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + threshold (int): [2, 5], dead threshold, default=3

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.human_based import BRO    >>> import numpy as np
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
    >>> model = BRO.DevBRO(epoch=1000, pop_size=50, threshold = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        threshold: float = 3,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            threshold (int): dead threshold, default=3
        """
        super().__init__(parameters=["epoch", "pop_size", "threshold"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=DevBROPopulation)
        self.threshold = cy.validator(float, threshold, [1, 10], "threshold")

    def initialize_variables(self):
        shrink = np.ceil(np.log10(self.epoch))
        self.dyn_delta = np.round(self.epoch / shrink)
        self.lb_updated = self.problem.bounds.low.copy()
        self.ub_updated = self.problem.bounds.up.copy()

    def get_idx_min__(self, data):
        k_zero = np.count_nonzero(data == 0)
        if k_zero == len(data):
            return self.generator.choice(range(0, k_zero))
        ## 1st: Partition sorting, not good solution here.
        # return np.argpartition(data, k_zero)[k_zero]
        ## 2nd: Faster
        return np.where(data == np.min(data[data != 0]))[0][0]

    def find_idx_min_distance__(self, target_pos=None, pop=None):
        pop_size = self.population.size()
        list_pos = np.array([pop[idx].solution for idx in range(0, pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, "euclidean")
        dist_list = np.reshape(dist_list, (-1))
        return self.get_idx_min__(dist_list)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx, agent in enumerate(self.population.toarray()):
            # Compare ith soldier with nearest one (jth)
            jdx = self.find_idx_min_distance__(agent.solution, self.population)
            if cy.is_better(agent, self.population[jdx], self.problem.sense):
                ## Update Winner based on global best solution
                x = agent.solution + self.generator.normal(
                    0, 1
                ) * np.mean(
                    np.array([agent.solution, self.g_best.solution]), axis=0
                )
                x = cy.correct_solution(self.problem, x)
                child = self.population.generate_agent(x)
                dam_new = (
                    agent.damage - 1
                )  ## Substract damaged hurt -1 to go next battle
                child.damage = dam_new
                self.population[idx] = child
                ## Update Loser
                if (
                    self.population[jdx].damage < self.threshold
                ):  ## If loser not dead yet, move it based on general
                    x = self.generator.uniform() * (
                        np.maximum(self.population[jdx].solution, self.g_best.solution)
                        - np.minimum(self.population[jdx].solution, self.g_best.solution)
                    ) + np.maximum(self.population[jdx].solution, self.g_best.solution)
                    dam_new = self.population[jdx].damage + 1
                    self.population[jdx].evaluate(self.problem)
                else:  ## Loser dead and respawn again
                    x = self.generator.uniform(
                        self.lb_updated, self.ub_updated
                    )
                    dam_new = 0
                x = cy.correct_solution(self.problem, x)
                child = self.population.generate_agent(x)
                child.damage = dam_new
                self.population[jdx] = child
            else:
                ## Update Loser by following position of Winner
                self.population[idx] = cy.duplicate_agent(self.population[jdx])
                ## Update Winner by following position of General to protect the King and General
                x = self.population[jdx].solution + self.generator.uniform() * (
                    self.g_best.solution - self.population[jdx].solution
                )
                x = cy.correct_solution(self.problem, x)
                child = self.population.generate_agent(x)
                child.damage = 0
                self.population[jdx] = child
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array(
                [self.population[idx].solution for idx in range(0, pop_size)]
            )
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best.solution - pos_std
            ub = self.g_best.solution + pos_std
            self.lb_updated = np.clip(
                lb, self.lb_updated, self.ub_updated
            )
            self.ub_updated = np.clip(
                ub, self.lb_updated, self.ub_updated
            )
            self.dyn_delta += np.round(self.dyn_delta / 2)
