#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 11:59, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from scipy.spatial.distance import cdist

from clypto.optimizer.native.agent cimport LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalSSpiderA(AgentListOptimizer):
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
    >>> from clypto.native.collection.vectorize.swarm_based import SSpiderA    >>> import numpy as np
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
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] James, J.Q. and Li, V.O., 2015. A social spider algorithm for global optimization. Applied soft computing, 30, pp.614-627.
    """

    cdef public object r_a
    cdef public object p_c
    cdef public object p_m

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r_a: float = 1.0,
        p_c: float = 0.7,
        p_m: float = 0.1,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_a (float): the rate of vibration attenuation when propagating over the spider web, default=1.0
            p_c (float): controls the probability of the spiders changing their dimension mask in the random walk step, default=0.7
            p_m (float): the probability of each value in a dimension mask to be one, default=0.1
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "r_a", "p_c", "p_m"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.r_a = cy.validator(float, r_a, (0, 5.0), "r_a")
        self.p_c = cy.validator(float, p_c, (0, 1.0), "p_c")
        self.p_m = cy.validator(float, p_m, (0, 1.0), "p_m")

    def _generate_empty_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        target_solution = solution.copy()
        local_vector = np.zeros(self.problem.n_dims)
        mask = np.zeros(self.problem.n_dims)
        return FieldAgent(
            solution=solution,
            target_solution=target_solution,
            local_vector=local_vector,
            mask=mask,
        )

    def _generate_agent(self, solution: np.ndarray | None = None) -> LegacyAgent:
        agent = self._generate_empty_agent(solution)
        agent.target = self._get_target(agent.solution)
        agent.intensity = np.log(
            1.0 / (np.abs(agent.target.fitness) + self.EPSILON) + 1
        )
        return agent

    def _evolve_agents(self, epoch):
        all_pos = np.array(
            [agent.solution for agent in self.objs]
        )  ## Matrix (pop_size, problem_size)
        base_distance = np.mean(np.std(all_pos, axis=0))  ## Number
        dist = cdist(all_pos, all_pos, "euclidean")
        intensity_source = np.array([it.intensity for it in self.objs])
        intensity_attenuation = np.exp(
            -dist / (base_distance * self.r_a)
        )  ## vector (pop_size)
        intensity_receive = np.dot(
            np.reshape(intensity_source, (1, self.pop_size)), intensity_attenuation
        )  ## vector (1, pop_size)
        id_best_intensity = np.argmax(intensity_receive)
        pop_new = []
        for idx in range(0, self.pop_size):
            agent = self.objs[idx].copy()
            if self.objs[id_best_intensity].intensity > self.objs[idx].intensity:
                agent.target_solution = self.objs[id_best_intensity].target_solution
            if self.generator.uniform() > self.p_c:  ## changing mask
                agent.mask = np.where(
                    self.generator.uniform(0, 1, self.problem.n_dims) < self.p_m, 0, 1
                )
            pos_new = np.where(
                self.objs[idx].mask == 0,
                self.objs[idx].target_solution,
                self.objs[self.generator.integers(0, self.pop_size)].solution,
            )
            ## Perform random walk
            pos_new = (
                self.objs[idx].solution
                + self.generator.normal()
                * (self.objs[idx].solution - self.objs[idx].local_vector)
                + (pos_new - self.objs[idx].solution) * self.generator.normal()
            )
            agent.solution = self._correct_solution(pos_new)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(agent.solution)
                agent.intensity = np.log(
                    1.0 / (np.abs(agent.target.fitness) + self.EPSILON) + 1
                )
                pop_new.append(agent)
        pop_new = self._update_target_for_population(pop_new)

        for idx in range(0, self.pop_size):
            if self._compare_target(
                pop_new[idx].target, self.objs[idx].target, self.problem.sense
            ):
                self.objs[idx].local_vector = (
                    pop_new[idx].solution - self.objs[idx].solution
                )
                self.objs[idx].intensity = np.log(
                    1.0 / (np.abs(pop_new[idx].target.fitness) + self.EPSILON) + 1
                )
                self.objs[idx].solution = pop_new[idx].solution
                self.objs[idx].target = pop_new[idx].target
