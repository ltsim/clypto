#!/usr/bin/env python
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.legacy.human_based.QSA.DevQSA cimport DevQSA
cimport clypto.core as cy


cdef class LevyQSA(DevQSA):
    """
    The Levy-flight version: Queuing Search Algorithm (LQSA)

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
    >>> model = QSA.LevyQSA(epoch=1000, pop_size=50)
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
        super().__init__(epoch, pop_size, **kwargs)
        self.sort_flag = True

    def update_business_2__(self, pop=None, current_epoch=None):
        pop_size = self.population.size()
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].fitness, pop[1].fitness, pop[2].fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        pr = [idx / pop_size for idx in range(1, pop_size + 1)]
        if t1 > 1.0e-6:
            cv = t1 / (t2 + t3)
        else:
            cv = 1 / 2
        pop_new = []
        for idx in range(pop_size):
            if idx < q1:
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                A = A2.copy()
            else:
                A = A3.copy()
            if self.generator.random() < pr[idx]:
                id1 = self.generator.choice(pop_size)
                if self.generator.random() < cv:
                    levy_step = cy.levy_flight(self.generator, beta=1.0, multiplier=0.001, size=None, case=-1)
                    X_new = (
                            pop[idx].solution
                            + self.generator.normal(0, 1, self.problem.n_dims) * levy_step
                    )
                else:
                    X_new = pop[idx].solution + self.generator.exponential(0.5) * (
                            A - pop[id1].solution
                    )
                pos_new = self.population.correct_solution(X_new)
            else:
                pos_new = self.problem.generate_solution()
            pos_new = self.population.correct_solution(pos_new)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return cy.sort_agents(pop_new, self.problem.sense)[:pop_size]

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = self.update_business_1__(self.population, epoch)
        pop = self.update_business_2__(pop, epoch)
        self.population = self.update_business_3__(pop, self.g_best)
