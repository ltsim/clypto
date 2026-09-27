#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class DevQSA(cy.Optimizer):
    """
    The developed version: Queuing Search Algorithm (QSA)

    Notes:
        + The third loops are removed
        + Global best solution is used in business 3-th instead of random solution

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.human_based import QSA    >>> import numpy as np
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
    >>> model = QSA.DevQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)

    def calculate_queue_length__(self, t1, t2, t3):
        """
        Calculate length of each queue based on  t1, t2,t3
            + t1 = t1 * 1.0e+100
            + t2 = t2 * 1.0e+100
            + t3 = t3 * 1.0e+100
        """
        pop_size = self.population.size()
        if t1 > 1.0e-6:
            n1 = (1 / t1) / ((1 / t1) + (1 / t2) + (1 / t3))
            n2 = (1 / t2) / ((1 / t1) + (1 / t2) + (1 / t3))
        else:
            n1 = 1.0 / 3
            n2 = 1.0 / 3
        q1 = int(n1 * pop_size)
        q2 = int(n2 * pop_size)
        q3 = pop_size - q1 - q2
        return q1, q2, q3

    def update_business_1__(self, pop=None, current_epoch=None):
        pop_size = self.population.size()
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].fitness, pop[1].fitness, pop[2].fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        case = None
        for idx in range(pop_size):
            if idx < q1:
                if idx == 0:
                    case = 1
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                if idx == q1:
                    case = 1
                A = A2.copy()
            else:
                if idx == q1 + q2:
                    case = 1
                A = A3.copy()
            beta = np.power(current_epoch, np.power(current_epoch / self.epoch, 0.5))
            alpha = self.generator.uniform(-1, 1)
            E = self.generator.exponential(0.5, self.problem.n_dims)
            F1 = beta * alpha * (
                    E * np.abs(A - pop[idx].solution)
            ) + self.generator.exponential(0.5) * (A - pop[idx].solution)
            F2 = beta * alpha * (E * np.abs(A - pop[idx].solution))
            if case == 1:
                pos_new = A + F1
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.generate_agent(pos_new)
                if cy.is_better(agent, pop[idx], self.problem.sense):
                    pop[idx] = agent
                else:
                    case = 2
            else:
                pos_new = pop[idx].solution + F2
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.generate_agent(pos_new)
                if cy.is_better(agent, pop[idx], self.problem.sense):
                    pop[idx] = agent
                else:
                    case = 1
        return cy.sort_agents(pop, self.problem.sense)

    def update_business_2__(self, pop=None):
        pop_size = self.population.size()
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].fitness, pop[1].fitness, pop[2].fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        pr = [idx / pop_size for idx in range(1, pop_size + 1)]
        if t1 > 1.0e-005:
            cv = t1 / (t2 + t3)
        else:
            cv = 1.0 / 2
        pop_new = []
        for idx in range(pop_size):
            if idx < q1:
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                A = A2.copy()
            else:
                A = A3.copy()
            if self.generator.random() < pr[idx]:
                i1, i2 = self.generator.choice(pop_size, 2, replace=False)
                if self.generator.random() < cv:
                    X_new = pop[idx].solution + self.generator.exponential(0.5) * (
                            pop[i1].solution - pop[i2].solution
                    )
                else:
                    X_new = pop[idx].solution + self.generator.exponential(0.5) * (
                            A - pop[i1].solution
                    )
            else:
                X_new = self.problem.generate_solution()
            pos_new = self.population.correct_solution(X_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return cy.sort_agents(pop_new, self.problem.sense)[:pop_size]

    def update_business_3__(self, pop, g_best):
        pop_size = self.population.size()
        pr = np.array([idx / pop_size for idx in range(1, pop_size + 1)])
        pop_new = []
        for idx in range(pop_size):
            X_new = pop[idx].solution.copy()
            id1 = self.generator.choice(pop_size)
            temp = g_best.solution + self.generator.exponential(
                0.5, self.problem.n_dims
            ) * (pop[id1].solution - pop[idx].solution)
            X_new = np.where(
                self.generator.random(self.problem.n_dims) > pr[idx], temp, X_new
            )
            pos_new = self.population.correct_solution(X_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return pop_new

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.population, epoch)
        pop = self.update_business_2__(pop)
        self.population = self.update_business_3__(pop, self.g_best)
