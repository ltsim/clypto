#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.swarm_based.PSO._base cimport PSOAgent, PSOPopulation


cdef class AIW_PSO(cy.Optimizer):
    """
    The original version of: Adaptive Inertia Weight Particle Swarm Optimization (AIW-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c1 (float): [1, 3], local coefficient, default = 2.05
        + c2 (float): [1, 3], global coefficient, default = 2.05
        + alpha (float): [0., 1.0], The positive constant, default = 0.4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import PSO    >>> import numpy as np
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
    >>> model = PSO.AIW_PSO(epoch=1000, pop_size=50, c1=2.05, c2=20.5, alpha=0.4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Qin, Z., Yu, F., Shi, Z., Wang, Y. (2006). Adaptive Inertia Weight Particle Swarm Optimization. In: Rutkowski, L.,
    Tadeusiewicz, R., Zadeh, L.A., Żurada, J.M. (eds) Artificial Intelligence and Soft Computing – ICAISC 2006. ICAISC 2006.
    Lecture Notes in Computer Science(), vol 4029. Springer, Berlin, Heidelberg. https://doi.org/10.1007/11785231_48
    """

    cdef public double alpha
    cdef public double c1
    cdef public double c2

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c1: float = 2.05,
        c2: float = 2.05,
        alpha: float = 0.4,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c1: [0-2] local coefficient
            c2: [0-2] global coefficient
            alpha: The positive constant, default = 0.4
        """
        super().__init__(**kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=PSOPopulation)
        self.c1 = cy.validator(float, c1, (0, 5.0), "c1")
        self.c2 = cy.validator(float, c2, (0, 5.0), "c2")
        self.alpha = cy.validator(float, alpha, [0.0, 1.0], "alpha")
        self.parameters = ["epoch", "pop_size", "c1", "c2", "alpha"]
        self.sort_flag = False

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        current_best = cy.duplicate_agent(self.population.sort()[0])
        cdef PSOAgent agent
        # sequential on purpose: g_best is one of the particles, and update_solution moves it in place,
        # so the particles after it already follow the new g_best (a batch step would change results)
        for agent in self.population.toarray():
            denom = np.abs(agent.pbest_solution - current_best.solution)
            denom = np.where(denom == 0, 1e-6, denom)
            isa = np.abs(agent.solution - agent.pbest_solution) / denom  # individual search ability
            w = 1 - self.alpha * (1.0 / (1.0 + np.exp(-isa)))
            agent.update_velocity(self.g_best.solution, w, self.c1, self.c2, self.generator,
                                  self.population.v_min, self.population.v_max)
            candidate = self.population.evaluate_solution(cy.reset_solution(self.problem, self.generator, agent.move()))
            if cy.is_better(candidate, agent, self.problem.sense):
                agent.update_solution(candidate)
            agent.update_pbest(candidate, self.problem.sense)
