#!/usr/bin/env python
# Created by "Thieu" at 13:59, 24/06/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy
from clypto.optimizer.native.agent cimport Agent
from clypto.optimizer.native.factory cimport AgentFactory





cdef class OriginalCOAAgents(AgentFactory):
    """The agent factory of ``OriginalCOA`` (moved off the optimizer)."""
    cdef Agent create(self, object solution=None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        age = 1

        agent = cy.Agent(solution)
        agent.age = age
        return agent

    cdef Agent generate(self, object solution=None):
        agent = self.create(solution)
        agent.evaluate(self.problem)
        return agent


cdef class OriginalCOA(cy.Optimizer):
    """
    The original version of: Coyote Optimization Algorithm (COA)

    Links:
        1. https://ieeexplore.ieee.org/document/8477769
        2. https://github.com/jkpir/COA/blob/master/COA.py  (Old version Mealpy < 1.2.2)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + n_coyotes (int): [3, 15], number of coyotes per group, default=5

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import COA    >>> import numpy as np
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
    >>> model = COA.OriginalCOA(epoch=1000, pop_size=50, n_coyotes = 5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Pierezan, J. and Coelho, L.D.S., 2018, July. Coyote optimization algorithm: a new metaheuristic
    for global optimization problems. In 2018 IEEE congress on evolutionary computation (CEC) (pp. 1-8). IEEE.
    """

    cdef public int n_coyotes

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        n_coyotes: int = 5,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            n_coyotes (int): number of coyotes per group, default=5
        """
        super().__init__(parameters=["epoch", "pop_size", "n_coyotes"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000])
        self.population.factory = OriginalCOAAgents()
        if n_coyotes == 5 and pop_size < 10:
            n_coyotes = max(2, int(pop_size / 2))
        self.n_coyotes = cy.validator(int, n_coyotes, [2, int(self.population.size() / 2)], "n_coyotes")
        self.n_packs = int(pop_size / self.n_coyotes)

    def initialization(self):
        pop_size = self.population.size()
        if len(self.population) == 0:
            self.population = self.population.spawn([self.population.generate_agent() for _ in range(pop_size)])
        base = pop_size // self.n_packs
        rem = pop_size % self.n_packs
        start = 0
        self.pop_group = []
        for p in range(self.n_packs):
            p_size = base + (1 if p < rem else 0)
            self.pop_group.append([cy.duplicate_agent(a) for a in self.population[start:start + p_size]])
            start += p_size
        self.ps = 1.0 / self.problem.n_dims
        self.p_leave = 0.005 * (self.n_coyotes**2)  # Probability of leaving a pack

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Execute the operations inside each pack
        for p in range(self.n_packs):
            pack_len = len(self.pop_group[p])
            # Get the coyotes that belong to each pack
            self.pop_group[p] = cy.sort_agents(self.pop_group[p], self.problem.sense)
            # Detect alphas according to the costs (Eq. 5)
            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.mean([agent.solution for agent in self.pop_group[p]])

            #  Update coyotes' social condition
            pop_new = []
            for i in range(pack_len):
                rc1, rc2 = self.generator.choice(
                    list(set(range(0, pack_len)) - {i}), 2, replace=False
                )
                # Try to update the social condition according to the alpha and the pack tendency(Eq. 12)
                x = (
                    self.pop_group[p][i].solution
                    + self.generator.random()
                    * (self.pop_group[p][0].solution - self.pop_group[p][rc1].solution)
                    + self.generator.random()
                    * (tendency - self.pop_group[p][rc2].solution)
                )
                # Keep the coyotes in the search space (optimization problem constraint)
                x = cy.correct_solution(self.problem, x)
                agent = self.population.create_agent(x)
                agent.age = self.pop_group[p][i].age
                pop_new.append(agent)
            # Evaluate the new social condition (Eq. 13)
            pop_new = self.population.evaluate(pop_new, self.mode)
            # Adaptation (Eq. 14)
            self.pop_group[p] = cy.greedy_agents(self.pop_group[p], pop_new, self.problem.sense)

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            id_dad, id_mom = self.generator.choice(
                list(range(0, pack_len)), 2, replace=False
            )
            prob1 = (1.0 - self.ps) / 2.0
            # Generate the pup considering intrinsic and extrinsic influence
            pup = np.where(
                self.generator.random(self.problem.n_dims) < prob1,
                self.pop_group[p][id_dad].solution,
                self.pop_group[p][id_mom].solution,
            )
            # Eventual noise
            x = self.generator.normal(0, 1) * pup
            x = cy.correct_solution(self.problem, x)
            agent = self.population.generate_agent(x)

            # Verify if the pup will survive
            packs = cy.sort_agents(self.pop_group[p], self.problem.sense)
            # Find index of element has fitness larger than new child. If existed an element like that, new child is good
            if cy.is_better(agent, packs[-1], self.problem.sense):
                if self.problem.sense == "min":
                    packs = sorted(packs, key=lambda agent: agent.age)
                else:
                    packs = sorted(packs, key=lambda agent: agent.age, reverse=True)
                # Replace worst element by new child, New born child with age = 0
                packs[-1] = agent
                self.pop_group[p] = [cy.duplicate_agent(agent) for agent in packs]

        # A coyote can leave a pack and enter in another pack (Eq. 4)
        if self.n_packs > 1:
            if self.generator.random() < self.p_leave:
                id_pack1, id_pack2 = self.generator.choice(
                    list(range(0, self.n_packs)), 2, replace=False
                )
                id1 = self.generator.integers(0, len(self.pop_group[id_pack1]))
                id2 = self.generator.integers(0, len(self.pop_group[id_pack2]))
                self.pop_group[id_pack1][id1], self.pop_group[id_pack2][id2] = (
                    self.pop_group[id_pack2][id2],
                    self.pop_group[id_pack1][id1],
                )

        # Update coyotes ages
        for id_pack in range(0, self.n_packs):
            for id_coy in range(0, len(self.pop_group[id_pack])):
                self.pop_group[id_pack][id_coy].age += 1
        self.population = self.population.spawn([agent for pack in self.pop_group for agent in pack])


