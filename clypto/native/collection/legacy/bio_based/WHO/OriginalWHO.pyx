#!/usr/bin/env python
# Created by "Thieu" at 12:51, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalWHO(cy.Optimizer):
    """
    The original version of: Wildebeest Herd Optimization (WHO)

    Links:
        1. https://doi.org/10.3233/JIFS-190495

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_explore_step (int): [2, 10] -> better [2, 4], number of exploration step
        + n_exploit_step (int): [2, 10] -> better [2, 4], number of exploitation step
        + eta (float): (0, 1.0) -> better [0.05, 0.5], learning rate
        + p_hi (float): (0, 1.0) -> better [0.7, 0.95], the probability of wildebeest move to another position based on herd instinct
        + local_alpha (float): (0, 3.0) -> better [0.5, 0.9], control local movement (alpha 1)
        + local_beta (float): (0, 3.0) -> better [0.1, 0.5], control local movement (beta 1)
        + global_alpha (float): (0, 3.0) -> better [0.1, 0.5], control global movement (alpha 2)
        + global_beta (float): (0, 3.0), control global movement (beta 2)
        + delta_w (float): (0.5, 5.0) -> better [1.0, 2.0], dist to worst
        + delta_c (float): (0.5, 5.0) -> better [1.0, 2.0], dist to best

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.bio_based import WHO    >>> import numpy as np
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
    >>> model = WHO.OriginalWHO(epoch=1000, pop_size=50, n_explore_step = 3, n_exploit_step = 3, eta = 0.15, p_hi = 0.9,
    >>>                         local_alpha=0.9, local_beta=0.3, global_alpha=0.2, global_beta=0.8, delta_w=2.0, delta_c=2.0)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Amali, D. and Dinakaran, M., 2019. Wildebeest herd optimization: a new global optimization algorithm inspired
    by wildebeest herding behaviour. Journal of Intelligent & Fuzzy Systems, 37(6), pp.8063-8076.
    """

    cdef public double delta_c
    cdef public double delta_w
    cdef public double eta
    cdef public double global_alpha
    cdef public double global_beta
    cdef public double local_alpha
    cdef public double local_beta
    cdef public int n_exploit_step
    cdef public int n_explore_step
    cdef public double p_hi

    def __init__(
            self,
            epoch=10000,
            pop_size=100,
            n_explore_step=3,
            n_exploit_step=3,
            eta=0.15,
            p_hi=0.9,
            local_alpha=0.9,
            local_beta=0.3,
            global_alpha=0.2,
            global_beta=0.8,
            delta_w=2.0,
            delta_c=2.0,
            **kwargs
    ):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_explore_step (int): default = 3, number of exploration step
            n_exploit_step (int): default = 3, number of exploitation step
            eta (float): default = 0.15, learning rate
            p_hi (float): default = 0.9, the probability of wildebeest move to another position based on herd instinct
            local_alpha (float): control local movement (alpha 1)
            local_beta (float): control local movement (beta 1)
            global_alpha (float): control global movement (alpha 2)
            global_beta (float): control global movement (beta 2)
            delta_w (float): dist to worst
            delta_c (float): dist to best
        """
        super().__init__(parameters=[ "epoch", "pop_size", "n_explore_step", "n_exploit_step", "eta", "p_hi", "local_alpha", "local_beta", "global_alpha", "global_beta", "delta_w", "delta_c", ], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.n_explore_step = cy.validator(int, n_explore_step, [2, 10], "n_explore_step")
        self.n_exploit_step = cy.validator(int, n_exploit_step, [2, 10], "n_exploit_step")
        self.eta = cy.validator(float, eta, (0, 1.0), "eta")
        self.p_hi = cy.validator(float, p_hi, (0, 1.0), "p_hi")
        self.local_alpha = cy.validator(float, local_alpha, (0, 3.0), "local_alpha")
        self.local_beta = cy.validator(float, local_beta, (0, 3.0), "local_beta")
        self.global_alpha = cy.validator(float, global_alpha, (0, 3.0), "global_alpha")
        self.global_beta = cy.validator(float, global_beta, (0, 3.0), "global_beta")
        self.delta_w = cy.validator(float, delta_w, (0.5, 5.0), "delta_w")
        self.delta_c = cy.validator(float, delta_c, (0.5, 5.0), "delta_c")

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Begin the Wildebeest Herd Optimization process
        pop_new = []
        for idx in range(0, pop_size):
            ### 1. Local movement (Milling behaviour)
            local_list = []
            for j in range(0, self.n_explore_step):
                temp = self.population[
                           idx
                       ].solution + self.eta * self.generator.uniform() * self.generator.uniform(
                    self.problem.bounds.low, self.problem.bounds.up
                )
                pos_new = self.population.correct_solution(temp)
                agent = self.population.create_agent(pos_new)
                local_list.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    local_list[-1].evaluate(self.problem)
            local_list = self.population.evaluate(local_list, self.mode)
            best_local = cy.sort_agents(local_list, self.problem.sense)[0].copy()
            temp = self.local_alpha * best_local.solution + self.local_beta * (
                    self.population[idx].solution - best_local.solution
            )
            pos_new = self.population.correct_solution(temp)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.evaluate(self.problem)
                self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self.population.evaluate(pop_new, self.mode)
            self.population = self.population.greedy(pop_new)
        for idx in range(0, pop_size):
            ### 2. Herd instinct
            idr = self.generator.choice(range(0, pop_size))
            if (
                    cy.is_better(self.population[idr], self.population[idx], self.problem.sense)
                    and self.generator.random() < self.p_hi
            ):
                temp = (
                        self.global_alpha * self.population[idx].solution
                        + self.global_beta * self.population[idr].solution
                )
                pos_new = self.population.correct_solution(temp)
                tar_new = self.population.evaluate_solution(pos_new)
                if cy.is_better(tar_new, self.population[idx], self.problem.sense):
                    self.population[idx].update_solution(tar_new, pos_new)

        ranked = self.population.sort()
        best = [agent.copy() for agent in ranked[:1]]
        worst = [agent.copy() for agent in ranked[::-1][:1]]
        g_best, g_worst = best[0], worst[0]
        pop_child = []
        for idx in range(0, pop_size):
            dist_to_worst = np.linalg.norm(self.population[idx].solution - g_worst.solution)
            dist_to_best = np.linalg.norm(self.population[idx].solution - g_best.solution)
            ### 3. Starvation avoidance
            if dist_to_worst < self.delta_w:
                temp = self.population[idx].solution + self.generator.uniform() * (
                        self.problem.bounds.up - self.problem.bounds.low
                ) * self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
                pos_new = self.population.correct_solution(temp)
                agent = self.population.create_agent(pos_new)
                pop_child.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
            ### 4. Population pressure
            if 1.0 < dist_to_best and dist_to_best < self.delta_c:
                temp = g_best.solution + self.eta * self.generator.uniform(
                    self.problem.bounds.low, self.problem.bounds.up
                )
                pos_new = self.population.correct_solution(temp)
                agent = self.population.create_agent(pos_new)
                pop_child.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
            ### 5. Herd social memory
            for jdx in range(0, self.n_exploit_step):
                temp = g_best.solution + 0.1 * self.generator.uniform(
                    self.problem.bounds.low, self.problem.bounds.up
                )
                pos_new = self.population.correct_solution(temp)
                agent = self.population.create_agent(temp)
                pop_child.append(agent)
                if self.mode not in self.AVAILABLE_MODES:
                    agent.evaluate(self.problem)
                    self.population[idx] = cy.get_better_agent(agent, self.population[idx], self.problem.sense)
        if self.mode in self.AVAILABLE_MODES:
            pop_child = self.population.evaluate(pop_child, self.mode)
            pop_child = cy.sort_agents(pop_child, self.problem.sense)[:pop_size]
            self.population = self.population.greedy(pop_child)
