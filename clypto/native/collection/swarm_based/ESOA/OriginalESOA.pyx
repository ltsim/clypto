#!/usr/bin/env python
# Created by "Thieu" at 17:48, 21/05/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy
from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory





cdef class OriginalESOAAgents(AgentFactory):
    """The agent factory of ``OriginalESOA`` (moved off the optimizer)."""
    cdef Agent create(self, object solution=None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        model_weights = self.generator.uniform(-1.0, 1.0, self.problem.n_dims)
        m = np.zeros(self.problem.n_dims)
        v = np.zeros(self.problem.n_dims)

        agent = cy.Agent(solution)
        agent.model_weights = model_weights
        agent.local_solution = solution.copy()
        agent.m = m
        agent.v = v
        return agent

    cdef Agent generate(self, object solution=None):
        agent = self.create(solution)
        agent.evaluate(self.problem)
        agent.local_best = cy.duplicate_agent(agent)
        agent.g = (np.sum(agent.model_weights * agent.solution) - agent.fitness) * agent.solution
        return agent


cdef class OriginalESOA(cy.Optimizer):
    """
    The original version of: Egret Swarm Optimization Algorithm (ESOA)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/115595-egret-swarm-optimization-algorithm-esoa
        2. https://www.mdpi.com/2313-7673/7/4/144

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import ESOA    >>> import numpy as np
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
    >>> model = ESOA.OriginalESOA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Chen, Z., Francis, A., Li, S., Liao, B., Xiao, D., Ha, T. T., ... & Cao, X. (2022). Egret Swarm Optimization Algorithm:
    An Evolutionary Computation Approach for Model Free Optimization. Biomimetics, 7(4), 144.
    """

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])
        self.population.factory = OriginalESOAAgents()

    def initialize_variables(self):
        self.beta1 = 0.9
        self.beta2 = 0.99

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        hop = self.problem.bounds.up - self.problem.bounds.low
        for idx, agent in enumerate(self.population.toarray()):
            # Individual Direction
            p_d = agent.local_solution - agent.solution
            p_d = p_d * (
                agent.local_best.fitness - agent.fitness
            )
            p_d = p_d / (np.sum(p_d) ** 2 + self.EPSILON)
            d_p = p_d + agent.g

            # Group Direction
            c_d = self.g_best.solution - agent.solution
            c_d = c_d * (self.g_best.fitness - agent.fitness)
            c_d = c_d / (np.sum(c_d) ** 2 + self.EPSILON)
            d_g = c_d + self.g_best.g

            # Gradient Estimation
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            g = (1 - r1 - r2) * agent.g + r1 * d_p + r2 * d_g
            g = g / (np.sum(g) + self.EPSILON)

            agent.m = self.beta1 * agent.m + (1 - self.beta1) * g
            agent.v = self.beta2 * agent.v + (1 - self.beta2) * g**2
            agent.model_weights -= agent.m / (
                np.sqrt(agent.v) + self.EPSILON
            )

            # Advice Forward
            x_0 = (
                agent.solution
                + np.exp(-1.0 / (0.1 * self.epoch)) * 0.1 * hop * g
            )
            x_0 = cy.correct_solution(self.problem, x_0)
            y_0 = self.population.evaluate_solution(x_0)

            # Random Search
            r3 = self.generator.uniform(-np.pi / 2, np.pi / 2, self.problem.n_dims)
            x_n = agent.solution + np.tan(r3) * hop / epoch * 0.5
            x_n = cy.correct_solution(self.problem, x_n)
            y_n = self.population.evaluate_solution(x_n)

            # Encircling Mechanism
            d = agent.local_solution - agent.solution
            d_g = self.g_best.solution - agent.solution
            r1 = self.generator.random(self.problem.n_dims)
            r2 = self.generator.random(self.problem.n_dims)
            x_m = (1 - r1 - r2) * agent.solution + r1 * d + r2 * d_g
            x_m = cy.correct_solution(self.problem, x_m)
            y_m = self.population.evaluate_solution(x_m)

            # Discriminant Condition
            y_list_compare = [y_0.fitness, y_n.fitness, y_m.fitness]
            y_list = [y_0, y_n, y_m]
            x_list = [x_0, x_n, x_m]
            if self.problem.sense == "min":
                id_best = np.argmin(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]
            else:
                id_best = np.argmax(y_list_compare)
                x_best = x_list[id_best]
                y_best = y_list[id_best]

            if cy.is_better(y_best, agent, self.problem.sense):
                agent.update_solution(y_best, x_best)
                if cy.is_better(y_best, agent.local_best, self.problem.sense):
                    agent.local_solution = x_best
                    agent.local_best = y_best
                    agent.g = (
                        np.sum(agent.model_weights * agent.solution)
                        - agent.fitness
                    ) * agent.solution
            else:
                if self.generator.random() < 0.3:
                    agent.update_solution(y_best, x_best)

