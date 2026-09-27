#!/usr/bin/env python
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class ABFOAgent(cy.Agent):
    cdef public object nutrients
    cdef public object local_solution
    cdef public object local_best


cdef class ABFOPopulation(cy.Population):
    """Agents of :class:`ABFO`."""

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        nutrients = 0  # total nutrient gained by the bacterium in its whole searching process.(int number)
        local_solution = solution.copy()
        return ABFOAgent(
            solution=solution, nutrients=nutrients, local_solution=local_solution
        )

    def generate_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        agent = self.create_agent(solution)
        agent.evaluate(self.problem)
        agent.local_best = agent.copy()
        return agent


cdef class ABFO(cy.Optimizer):
    """
    The original version of: Adaptive Bacterial Foraging Optimization (ABFO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:

        + C_s (float): step size start, default=0.1
        + C_e (float): step size end, default=0.001
        + Ped (float): Probability eliminate, default=0.01
        + Ns (int): swim_length, default=4
        + N_adapt (int): Dead threshold value default=2
        + N_split (int): Split threshold value, default=40

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BFO    >>> import numpy as np
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
    >>> model = BFO.ABFO(epoch=1000, pop_size=50, C_s=0.1, C_e=0.001, Ped = 0.01, Ns = 4, N_adapt = 2, N_split = 40)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Nguyen, B.M. and Nguyen, G., 2019, April. Building resource auto-scaler with functional-link
    neural network and adaptive bacterial foraging optimization. In International Conference on
    Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.
    """

    cdef public int N_adapt
    cdef public int N_split
    cdef public int Ns
    cdef public double Ped

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        C_s: float = 0.1,
        C_e: float = 0.001,
        Ped: float = 0.01,
        Ns: int = 4,
        N_adapt: int = 2,
        N_split: int = 40,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            C_s (float): step size start, default=0.1
            C_e (float): step size end, default=0.001
            Ped (float): Probability eliminate, default=0.01
            Ns (int): swim_length, default=4
            N_adapt (int): Dead threshold value default=2
            N_split (int): Split threshold value, default=40
        """
        super().__init__(parameters=["epoch", "pop_size", "C_s", "C_e", "Ped", "Ns", "N_adapt", "N_split"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=ABFOPopulation)
        self.C_s = self.Ped = cy.validator(float, C_s, (0, 2.0), "C_s")
        self.C_e = self.Ped = cy.validator(float, C_e, (0, 1.0), "C_e")
        self.p_eliminate = self.Ped = cy.validator(float, Ped, (0, 1.0), "Ped")
        self.swim_length = self.Ns = cy.validator(int, Ns, [2, 100], "Ns")
        self.N_adapt = cy.validator(int, N_adapt, [0, 4], "N_adapt")
        self.N_split = cy.validator(int, N_split, [5, 50], "N_split")
        self.support_parallel_modes = False

    def initialize_variables(self):
        self.C_s = self.C_s * (self.problem.bounds.up - self.problem.bounds.low)
        self.C_e = self.C_e * (self.problem.bounds.up - self.problem.bounds.low)

    def update_step_size__(self, pop=None, idx=None):
        total_fitness = np.sum([agent.fitness for agent in pop])
        step_size = (
            self.C_s - (self.C_s - self.C_e) * pop[idx].fitness / total_fitness
        )
        step_size = (
            step_size / self.population[idx].nutrients
            if self.population[idx].nutrients > 0
            else step_size
        )
        return step_size

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        for idx in range(0, pop_size):
            step_size = self.update_step_size__(self.population, idx)
            for m in range(0, self.swim_length):  # Ns
                delta_i = (self.g_best.solution - self.population[idx].solution) + (
                    self.population[idx].local_solution - self.population[idx].solution
                )
                delta = np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                unit_vector = (
                    self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
                    if delta == 0
                    else (delta_i / delta)
                )
                pos_new = self.population[idx].solution + step_size * unit_vector
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.generate_agent(pos_new)
                if cy.is_better(agent, self.population[idx], self.problem.sense):
                    agent.nutrients += 1
                    self.population[idx] = agent
                    # Update personal best
                    if cy.is_better(agent, self.population[idx].local_best, self.problem.sense):
                        self.population[idx].update(
                            local_solution=pos_new.copy(),
                            local_best=agent.copy(),
                        )
                else:
                    self.population[idx].nutrients -= 1
            if self.population[idx].nutrients > max(
                self.N_split,
                self.N_split + (len(self.population) - pop_size) / self.N_adapt,
            ):
                tt = self.generator.normal(0, 1, self.problem.n_dims)
                pos_new = tt * self.population[idx].solution + (1 - tt) * (
                    self.g_best.solution - self.population[idx].solution
                )
                pos_new = self.population.correct_solution(pos_new)
                agent = self.population.generate_agent(pos_new)
                self.population.append(agent)
            nut_min = min(
                self.N_adapt,
                self.N_adapt + (len(self.population) - pop_size) / self.N_adapt,
            )
            if (
                self.population[idx].nutrients < nut_min
                or self.generator.random() < self.p_eliminate
            ):
                self.population[idx] = self.population.generate_agent()
        ## Make sure the population does not have duplicates.
        new_set = set()
        for idx, obj in enumerate(self.population):
            if tuple(obj.solution.tolist()) in new_set:
                self.population.pop(idx)
            else:
                new_set.add(tuple(obj.solution.tolist()))
        ## Balance the population by adding more agents or remove some agents
        n_agents = len(self.population) - pop_size
        if n_agents < 0:
            for idx in range(0, n_agents):
                agent = self.population.generate_agent()
                self.population.append(agent)
        elif n_agents > 0:
            list_idx_removed = self.generator.choice(
                range(0, len(self.population)), n_agents, replace=False
            )
            pop_new = []
            for idx in range(0, len(self.population)):
                if idx not in list_idx_removed:
                    pop_new.append(self.population[idx])
            self.population = pop_new
