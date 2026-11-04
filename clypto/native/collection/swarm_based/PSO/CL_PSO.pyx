#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy

from clypto.native.collection.swarm_based.PSO._base cimport PSOPopulation


cdef class CL_PSO(cy.Optimizer):
    """
    The original version of: Comprehensive Learning Particle Swarm Optimization (CL-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_local (float): [1.0, 3.0], local coefficient, default = 1.2
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.7, 2.0], Weight max of bird, default = 0.9
        + max_flag (int): [5, 20], Number of times, default = 7

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
    >>> model = PSO.CL_PSO(epoch=1000, pop_size=50, c_local = 1.2, w_min=0.4, w_max=0.9, max_flag = 7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer
    for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
    """

    cdef public double c_local
    cdef public int max_flag
    cdef public double w_max
    cdef public double w_min

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c_local: float = 1.2,
        w_min: float = 0.4,
        w_max: float = 0.9,
        max_flag: int = 7,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c_local: local coefficient, default = 1.2
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
            max_flag: Number of times, default = 7
        """
        super().__init__(**kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=PSOPopulation)
        self.c_local = cy.validator(float, c_local, (0, 5.0), "c_local")
        self.w_min = cy.validator(float, w_min, (0, 0.5), "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")
        self.max_flag = cy.validator(int, max_flag, [2, 100], "max_flag")
        self.parameters = ["epoch", "pop_size", "c_local", "w_min", "w_max", "max_flag"]
        self.sort_flag = False

    def initialize_variables(self):
        pop_size = self.population.size()
        self.v_max = 0.5 * (self.problem.bounds.up - self.problem.bounds.low)
        self.v_min = -self.v_max
        self.flags = np.zeros(pop_size)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        wk = self.w_max * (epoch / self.epoch) * (self.w_max - self.w_min)
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            pci = 0.05 + 0.45 * (np.exp(10 * (idx + 1) / pop_size) - 1) / (
                np.exp(10) - 1
            )
            vec_new = self.population[idx].velocity.copy()
            for jdx in range(0, self.problem.n_dims):
                if self.generator.random() > pci:
                    vj = wk * self.population[idx].velocity[
                        jdx
                    ] + self.c_local * self.generator.random() * (
                        self.population[idx].pbest_solution[jdx] - self.population[idx].solution[jdx]
                    )
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, pop_size)) - {idx}), 2, replace=False
                    )
                    if cy.is_better(self.population[id1], self.population[id2], self.problem.sense):
                        vj = wk * self.population[idx].velocity[
                            jdx
                        ] + self.c_local * self.generator.random() * (
                            self.population[id1].pbest_solution[jdx]
                            - self.population[idx].solution[jdx]
                        )
                    else:
                        vj = wk * self.population[idx].velocity[
                            jdx
                        ] + self.c_local * self.generator.random() * (
                            self.population[id2].pbest_solution[jdx]
                            - self.population[idx].solution[jdx]
                        )
                vec_new[jdx] = vj
            vec_new = np.clip(vec_new, self.v_min, self.v_max)
            x = self.population[idx].solution + vec_new
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            n_population.append(agent)
            if self.mode == "sequential":
                agent.evaluate(self.problem)
                agent.update_pbest(agent, self.problem.sense)
                self.population[idx] = cy.get_better_agent(self.population[idx], agent, self.problem.sense)
                if self.population[idx].update_pbest(agent, self.problem.sense):
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1
                    if self.flags[idx] >= self.max_flag:
                        self.flags[idx] = 0
        if self.mode != "sequential":
            n_population = self.population.evaluate(n_population, self.mode)
            pop_child = self.population.greedy(n_population)
            for idx, agent in enumerate(self.population.toarray()):
                if cy.better_fitness(n_population[idx].fitness, agent.pbest_fitness, self.problem.sense):
                    pop_child[idx].pbest_solution = n_population[idx].solution.copy()
                    pop_child[idx].pbest_fitness = n_population[idx].fitness
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1
                    if self.flags[idx] >= self.max_flag:
                        self.flags[idx] = 0
            self.population = self.population.spawn(pop_child)
