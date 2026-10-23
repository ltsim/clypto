#!/usr/bin/env python
# Created by "Thieu" at 14:22, 11/04/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalMAAgent(cy.Agent):
    cdef public object bitstring

    def __init__(self, solution=None, objectives=None, weights=None, bitstring=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.bitstring = bitstring

    cdef cy.Agent clone(self):
        cdef OriginalMAAgent new = <OriginalMAAgent>cy.Agent.clone(self)
        new.bitstring = self.bitstring
        return new


cdef class OriginalMAPopulation(cy.Population):
    """Agents of :class:`OriginalMA`."""
    cdef public object bits_total

    cdef void copy_state(self, cy.Population new):
        cy.Population.copy_state(self, new)
        (<OriginalMAPopulation>new).bits_total = self.bits_total

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        bitstring = "".join(
            [
                "1" if self.generator.uniform() < 0.5 else "0"
                for _ in range(0, self.bits_total)
            ]
        )
        return OriginalMAAgent(solution=solution, bitstring=bitstring)


cdef class OriginalMA(cy.Optimizer):
    """
    The original version of: Memetic Algorithm (MA)

    Links:
        1. https://www.cleveralgorithms.com/nature-inspired/physical/memetic_algorithm.html
        2. https://github.com/clever-algorithms/CleverAlgorithms

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pc (float): [0.7, 0.95], cross-over probability, default = 0.85
        + pm (float): [0.05, 0.3], mutation probability, default = 0.15
        + p_local (float): [0.3, 0.7], Probability of local search for each agent, default=0.5
        + max_local_gens (int): [5, 25], number of local search agent will be created during local search mechanism, default=10
        + bits_per_param (int): [2, 4, 8, 16], number of bits to decode a real number to 0-1 bitstring, default=4

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.evolutionary_based import MA    >>> import numpy as np
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
    >>> model = MA.OriginalMA(epoch=1000, pop_size=50, pc = 0.85, pm = 0.15, p_local = 0.5, max_local_gens = 10, bits_per_param = 4)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Moscato, P., 1989. On evolution, search, optimization, genetic algorithms and martial arts:
    Towards memetic algorithms. Caltech concurrent computation program, C3P Report, 826, p.1989.
    """

    cdef public int bits_per_param
    cdef public int max_local_gens
    cdef public double p_local
    cdef public double pc
    cdef public double pm

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        pc: float = 0.85,
        pm: float = 0.15,
        p_local: float = 0.5,
        max_local_gens: int = 10,
        bits_per_param: int = 4,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            pc (float): cross-over probability, default = 0.85
            pm (float): mutation probability, default = 0.15
            p_local (float): Probability of local search for each agent, default=0.5
            max_local_gens (int): Number of local search agent will be created during local search mechanism, default=10
            bits_per_param (int): Number of bits to decode a real number to 0-1 bitstring, default=4
        """
        super().__init__(parameters=[ "epoch", "pop_size", "pc", "pm", "p_local", "max_local_gens", "bits_per_param", ], sort_flag=True, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalMAPopulation)
        self.pc = cy.validator(float, pc, (0, 1.0), "pc")
        self.pm = cy.validator(float, pm, (0, 1.0), "pm")
        self.p_local = cy.validator(float, p_local, (0, 1.0), "p_local")
        self.max_local_gens = cy.validator(int, max_local_gens, [2, int(pop_size / 2)], "max_local_gens")
        self.bits_per_param = cy.validator(int, bits_per_param, [2, 32], "bits_per_param")

    def initialize_variables(self):
        self.bits_total = self.problem.n_dims * self.bits_per_param
        self.population.bits_total = self.bits_total

    def decode__(self, bitstring: str | None = None) -> np.ndarray:
        """
        Decode the random bitstring into real number

        Args:
            bitstring (str): "11000000100101000101010" - bits_per_param = 16, 32 bit for 2 variable. eg. x1 and x2

        Returns:
            list: list of real number (vector)
        """
        vector = np.ones(self.problem.n_dims)
        for idx in range(0, self.problem.n_dims):
            param = bitstring[
                idx * self.bits_per_param : (idx + 1) * self.bits_per_param
            ]  # Select 16 bit every time
            vector[idx] = self.problem.bounds.low[idx] + (
                (self.problem.bounds.up[idx] - self.problem.bounds.low[idx])
                / ((2.0**self.bits_per_param) - 1)
            ) * int(param, 2)
        return vector

    def crossover__(self, dad=None, mom=None):
        if self.generator.uniform() >= self.pc:
            return dad
        else:
            child = ""
            for idx in range(0, self.bits_total):
                if self.generator.uniform() < 0.5:
                    child += dad[idx]
                else:
                    child += mom[idx]
            return child

    def point_mutation__(self, bitstring=None):
        child = ""
        for bit in bitstring:
            if self.generator.uniform() < self.pc:
                child += "0" if bit == "1" else "1"
            else:
                child += bit
        return child

    def bits_climber__(self, child=None):
        current = child
        list_local = []
        for idx in range(0, self.max_local_gens):
            child = current
            bitstring_new = self.point_mutation__(child.bitstring)
            x = self.decode__(bitstring_new)
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            agent.solution = x
            agent.bitstring = bitstring_new
            list_local.append(agent)
        list_local = self.population.evaluate(list_local, self.mode)
        list_local.append(child)
        best = cy.duplicate_agent(cy.sort_agents(list_local, self.problem.sense)[0])
        return best

    def create_child__(self, idx, pop_copy):
        pop_size = self.population.size()
        ancient = pop_copy[idx + 1] if idx % 2 == 0 else pop_copy[idx - 1]
        if idx == pop_size - 1:
            ancient = pop_copy[0]
        bitstring_new = self.crossover__(pop_copy[idx].bitstring, ancient.bitstring)
        bitstring_new = self.point_mutation__(bitstring_new)
        x = self.decode__(bitstring_new)
        x = cy.correct_solution(self.problem, x)
        agent = self.population.generate_agent(x)
        agent.bitstring = bitstring_new
        return agent

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        ## Binary tournament
        children = []
        for idx in range(0, pop_size):
            idx_offspring = cy.kway_tournament(self.generator, self.problem.sense, self.population, k_way=2, output=1)[0]
            children.append(cy.duplicate_agent(self.population[idx_offspring]))
        pop = []
        for idx in range(0, pop_size):
            if idx == pop_size - 1:
                ancient = children[0]
            else:
                ancient = children[idx + 1] if idx % 2 == 0 else children[idx - 1]
            bitstring_new = self.crossover__(children[idx].bitstring, ancient.bitstring)
            bitstring_new = self.point_mutation__(bitstring_new)
            x = self.decode__(bitstring_new)
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            agent.bitstring = bitstring_new
            pop.append(agent)
            if self.mode == "sequential":
                pop[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(pop, self.mode))
        # Searching in local
        for idx, agent in enumerate(self.population.toarray()):
            if self.generator.random() < self.p_local:
                self.population[idx] = self.bits_climber__(pop[idx])
