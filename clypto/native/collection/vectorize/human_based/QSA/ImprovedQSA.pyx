#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:21, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.native.collection.vectorize.human_based.QSA.DevQSA cimport DevQSA
cimport clypto.core as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class ImprovedQSA(DevQSA):
    """
    The original version of: Improved Queuing Search Algorithm (QSA)

    Links:
       1. https://doi.org/10.1007/s12652-020-02849-4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import QSA    >>> import numpy as np
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
    >>> model = QSA.ImprovedQSA(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, B.M., Hoang, B., Nguyen, T. and Nguyen, G., 2021. nQSV-Net: a novel queuing search variant for
    global space search and workload modeling. Journal of Ambient Intelligence and Humanized Computing, 12(1), pp.27-46.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(epoch, pop_size, name=name, mode=mode)
        self.sort_flag = True

    def update_business_2__(self, pop=None, current_epoch=None):
        A1, A2, A3 = pop[0].solution, pop[1].solution, pop[2].solution
        t1, t2, t3 = pop[0].fitness, pop[1].fitness, pop[2].fitness
        q1, q2, q3 = self.calculate_queue_length__(t1, t2, t3)
        pr = [idx / self.pop_size for idx in range(1, self.pop_size + 1)]
        if t1 > 1.0e-6:
            cv = t1 / (t2 + t3)
        else:
            cv = 1 / 2
        pop_new = []
        for idx in range(self.pop_size):
            if idx < q1:
                A = A1.copy()
            elif q1 <= idx < q1 + q2:
                A = A2.copy()
            else:
                A = A3.copy()
            if self.generator.random() < pr[idx]:
                id1 = self.generator.choice(self.pop_size)
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
                pos_new = self.correct_solution(X_new)
            else:
                pos_new = self.problem.generate_solution()
            pos_new = self.correct_solution(pos_new)
            agent = self.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.evaluate_agents(pop_new)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return cy.sort_agents(pop_new, self.problem.sense)[:self.pop_size]

    def opposition_based__(self, pop=None, g_best=None):
        pop = cy.sort_agents(pop, self.problem.sense)
        pop_new = []
        for idx in range(0, self.pop_size):
            # opposition-based candidate around g_best
            pos_new = self.correct_solution(
                self.problem.bounds.low + self.problem.bounds.up - g_best.solution
                + self.generator.uniform() * (g_best.solution - pop[idx].solution)
            )
            pos_new = self.correct_solution(pos_new)
            agent = self.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                pop_new[-1] = cy.get_better_agent(agent, pop[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.evaluate_agents(pop_new)
            pop_new = cy.greedy_agents(pop, pop_new, self.problem.sense)
        return pop_new

    def evolve_agents(self, epoch):
        pop = self.update_business_1__(self.objs, epoch)
        pop = self.update_business_2__(pop, epoch)
        pop = self.update_business_3__(pop, self.g_best)
        self.objs = self.opposition_based__(pop, self.g_best)
