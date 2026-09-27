#!/usr/bin/env python
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
from scipy.spatial.distance import cdist
cimport clypto.core as cy



cdef class OriginalSSpiderAAgent(cy.Agent):
    cdef public object target_solution
    cdef public object local_vector
    cdef public object mask
    cdef public object intensity

    def __init__(self, solution=None, objectives=None, weights=None, target_solution=None, local_vector=None, mask=None, intensity=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.target_solution = target_solution
        self.local_vector = local_vector
        self.mask = mask
        self.intensity = intensity

    cdef cy.Agent clone(self):
        cdef OriginalSSpiderAAgent new = <OriginalSSpiderAAgent>cy.Agent.clone(self)
        new.target_solution = self.target_solution
        new.local_vector = self.local_vector
        new.mask = self.mask
        new.intensity = self.intensity
        return new


cdef class OriginalSSpiderAPopulation(cy.Population):
    """Agents of :class:`OriginalSSpiderA`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        """
        Overriding method in LegacyOptimizer class
            + x: The position of s on the web.
            + train: The fitness of the current position of s
            + target_vibration: The target vibration of s in the previous iteration.
            + intensity_vibration: intensity of vibration
            + movement_vector: The movement that s performed in the previous iteration
            + dimension_mask: The dimension mask 1 that s employed to guide movement in the previous iteration
            + The dimension mask is a 0-1 binary vector of length problem size
            + n_changed: The number of iterations since s has last changed its target vibration. (No need)
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        target_solution = solution.copy()
        local_vector = np.zeros(self.problem.n_dims)
        mask = np.zeros(self.problem.n_dims)
        return OriginalSSpiderAAgent(
            solution=solution,
            target_solution=target_solution,
            local_vector=local_vector,
            mask=mask,
        )

    def generate_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        """
        Generate new agent with full information

        Args:
            solution (np.ndarray): The solution
        """
        agent = self.create_agent(solution)
        agent.evaluate(self.problem)
        agent.intensity = np.log(
            1.0 / (np.abs(agent.fitness) + 10e-10) + 1
        )
        return agent


cdef class OriginalSSpiderA(cy.Optimizer):
    """
    The developed version of: Social Spider Algorithm (OriginalSSpiderA)

    Notes:
        + The version of the algorithm available on the GitHub repository has a slow convergence rate.
        + Changes the idea of intensity, which one has better intensity, others will move toward to it
        + https://doi.org/10.1016/j.asoc.2015.02.014
        + https://github.com/James-Yu/SocialSpiderAlgorithm  (Modified this version)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
        + p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
        + p_m (float): the probability of each value in a dimension mask to be one, default=0.1

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import SSpiderA    >>> import numpy as np
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
    >>> model = SSpiderA.OriginalSSpiderA(epoch=1000, pop_size=50, r_a = 1.0, p_c = 0.7, p_m = 0.1)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] James, J.Q. and Li, V.O., 2015. A social spider algorithm for global optimization. Applied soft computing, 30, pp.614-627.
    """

    cdef public double p_c
    cdef public double p_m
    cdef public double r_a

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r_a: float = 1.0,
        p_c: float = 0.7,
        p_m: float = 0.1,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
            p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
            p_m (float): the probability of each value in a dimension mask to be one, default=0.1
        """
        super().__init__(parameters=["epoch", "pop_size", "r_a", "p_c", "p_m"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalSSpiderAPopulation)
        self.r_a = cy.validator(float, r_a, (0, 5.0), "r_a")
        self.p_c = cy.validator(float, p_c, (0, 1.0), "p_c")
        self.p_m = cy.validator(float, p_m, (0, 1.0), "p_m")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        all_pos = np.array(
            [agent.solution for agent in self.population]
        )  ## Matrix (pop_size, problem_size)
        base_distance = np.mean(np.std(all_pos, axis=0))  ## Number
        dist = cdist(all_pos, all_pos, "euclidean")
        intensity_source = np.array([it.intensity for it in self.population])
        intensity_attenuation = np.exp(
            -dist / (base_distance * self.r_a)
        )  ## vector (pop_size)
        intensity_receive = np.dot(
            np.reshape(intensity_source, (1, pop_size)), intensity_attenuation
        )  ## vector (1, pop_size)
        id_best_intensity = np.argmax(intensity_receive)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            agent = cy.duplicate_agent(self.population[idx])
            if self.population[id_best_intensity].intensity > self.population[idx].intensity:
                agent.target_solution = self.population[id_best_intensity].target_solution
            if self.generator.uniform() > self.p_c:  ## changing mask
                agent.mask = np.where(
                    self.generator.uniform(0, 1, self.problem.n_dims) < self.p_m, 0, 1
                )
            x = np.where(
                self.population[idx].mask == 0,
                self.population[idx].target_solution,
                self.population[self.generator.integers(0, pop_size)].solution,
            )
            ## Perform random walk
            x = (
                self.population[idx].solution
                + self.generator.normal()
                * (self.population[idx].solution - self.population[idx].local_vector)
                + (x - self.population[idx].solution) * self.generator.normal()
            )
            agent.solution = cy.correct_solution(self.problem, x)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                agent.intensity = np.log(
                    1.0 / (np.abs(agent.fitness) + self.EPSILON) + 1
                )
            n_population.append(agent)
        n_population = self.population.evaluate(n_population, self.mode)

        for idx, agent in enumerate(self.population.toarray()):
            if cy.is_better(n_population[idx], agent, self.problem.sense):
                agent.local_vector = (
                    n_population[idx].solution - agent.solution
                )
                agent.intensity = np.log(
                    1.0 / (np.abs(n_population[idx].fitness) + self.EPSILON) + 1
                )
                agent.update_solution(n_population[idx], n_population[idx].solution)
