#!/usr/bin/env python
# Created by "Thieu" at 10:09, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class OriginalCSOAgent(cy.Agent):
    cdef public object velocity
    cdef public object flag

    def __init__(self, solution=None, objectives=None, weights=None, velocity=None, flag=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.velocity = velocity
        self.flag = flag

    cdef cy.Agent clone(self):
        cdef OriginalCSOAgent new = <OriginalCSOAgent>cy.Agent.clone(self)
        new.velocity = self.velocity
        new.flag = self.flag
        return new


cdef class OriginalCSOPopulation(cy.Population):
    """Agents of :class:`OriginalCSO`."""
    cdef public object mixture_ratio

    cdef void copy_state(self, cy.Population new):
        cy.Population.copy_state(self, new)
        (<OriginalCSOPopulation>new).mixture_ratio = self.mixture_ratio

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        """
        + x: current position of cat
        + v: vector v of cat (same amount of dimension as x)
        + flag: the stage of cat, seeking (looking/finding around) or tracing (chasing/catching) => False: seeking mode , True: tracing mode
        """
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        flag = True if self.generator.uniform() < self.mixture_ratio else False
        return OriginalCSOAgent(solution=solution, velocity=velocity, flag=flag)


cdef class OriginalCSO(cy.Optimizer):
    """
    The original version of: Cat Swarm Optimization (CSO)

    Links:
        1. https://link.springer.com/chapter/10.1007/978-3-540-36668-3_94
        2. https://www.hindawi.com/journals/cin/2020/4854895/

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + mixture_ratio (float): joining seeking mode with tracing mode, default=0.15
        + smp (int): seeking memory pool, default=5 clones (larger is better but time-consuming)
        + spc (bool): self-position considering, default=False
        + cdc (float): counts of dimension to change (larger is more diversity but slow convergence), default=0.8
        + srd (float): seeking range of the selected dimension (smaller is better but slow convergence), default=0.15
        + c1 (float): same in PSO, default=0.4
        + w_min (float): same in PSO
        + w_max (float): same in PSO
        + selected_strategy (int):  0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import CSO    >>> import numpy as np
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
    >>> model = CSO.OriginalCSO(epoch=1000, pop_size=50, mixture_ratio = 0.15, smp = 5, spc = False, cdc = 0.8, srd = 0.15, c1 = 0.4, w_min = 0.4, w_max = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Chu, S.C., Tsai, P.W. and Pan, J.S., 2006, August. Cat swarm optimization. In Pacific Rim
    international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
    """

    cdef public double c1
    cdef public double cdc
    cdef public double mixture_ratio
    cdef public int selected_strategy
    cdef public int smp
    cdef public bint spc
    cdef public double srd
    cdef public double w_max
    cdef public double w_min

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        mixture_ratio: float = 0.15,
        smp: int = 5,
        spc: bool = False,
        cdc: float = 0.8,
        srd: float = 0.15,
        c1: float = 0.4,
        w_min: float = 0.5,
        w_max: float = 0.9,
        selected_strategy: int = 1,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            mixture_ratio (float): joining seeking mode with tracing mode
            smp (int): seeking memory pool, 10 clones  (larger is better but time-consuming)
            spc (bool): self-position considering
            cdc (float): counts of dimension to change  (larger is more diversity but slow convergence)
            srd (float): seeking range of the selected dimension (smaller is better but slow convergence)
            c1 (float): same in PSO
            w_min (float): same in PSO
            w_max (float): same in PSO
            selected_strategy (int):  0: best fitness, 1: tournament, 2: roulette wheel, else: random (decrease by quality)
        """
        super().__init__(parameters=[ "epoch", "pop_size", "mixture_ratio", "smp", "spc", "cdc", "srd", "c1", "w_min", "w_max", "selected_strategy", ], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=OriginalCSOPopulation)
        self.mixture_ratio = cy.validator(float, mixture_ratio, (0, 1.0), "mixture_ratio")
        self.smp = cy.validator(int, smp, [2, 10000], "smp")
        self.population.mixture_ratio = self.mixture_ratio
        self.spc = cy.validator(bool, spc, (True, False), "spc")
        self.cdc = cy.validator(float, cdc, (0, 1.0), "cdc")
        self.srd = cy.validator(float, srd, (0, 1.0), "srd")
        self.c1 = cy.validator(float, c1, (0, 3.0), "c1")
        self.w_min = cy.validator(float, w_min, [0.1, 0.5], "w_min")
        self.w_max = cy.validator(float, w_max, [0.5, 2.0], "w_max")
        self.selected_strategy = cy.validator(int, selected_strategy, [0, 4], "selected_strategy")

    def seeking_mode__(self, cat):
        candidate_cats = []
        clone_cats = self.population.generate(self.smp)
        if self.spc:
            candidate_cats.append(cy.duplicate_agent(cat))
            clone_cats = [cy.duplicate_agent(cat) for _ in range(self.smp - 1)]
        for clone in clone_cats:
            idx = self.generator.choice(
                range(0, self.problem.n_dims),
                int(self.cdc * self.problem.n_dims),
                replace=False,
            )
            pos_new1 = clone.solution * (1 + self.srd)
            pos_new2 = clone.solution * (1 - self.srd)
            x = np.where(
                self.generator.random(self.problem.n_dims) < 0.5, pos_new1, pos_new2
            )
            x[idx] = clone.solution[idx]
            x = cy.correct_solution(self.problem, x)
            agent = self.population.create_agent(x)
            agent.velocity = clone.velocity
            agent.flag = clone.flag
            candidate_cats.append(agent)
        candidate_cats = self.population.evaluate(candidate_cats, self.mode)

        if self.selected_strategy == 0:  # Best fitness-self
            cat = cy.duplicate_agent(cy.sort_agents(candidate_cats, self.problem.sense)[0])
        elif self.selected_strategy == 1:  # Tournament
            k_way = 4
            idx = self.generator.choice(range(0, self.smp), k_way, replace=False)
            cats_k_way = [candidate_cats[_] for _ in idx]
            cat = cy.duplicate_agent(cy.sort_agents(cats_k_way, self.problem.sense)[0])
        elif self.selected_strategy == 2:  ### Roul-wheel selection
            list_fitness = [
                candidate_cats[u].fitness for u in range(0, len(candidate_cats))
            ]
            idx = cy.roulette_wheel(self.generator, self.problem.sense, list_fitness)
            cat = candidate_cats[idx]
        else:
            idx = self.generator.choice(range(0, len(candidate_cats)))
            cat = candidate_cats[idx]  # Random
        return cat.solution

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            child = cy.duplicate_agent(agent)
            # tracing mode
            if agent.flag:
                x = (
                    agent.solution
                    + w * agent.velocity
                    + self.generator.uniform()
                    * self.c1
                    * (self.g_best.solution - agent.solution)
                )
                x = cy.correct_solution(self.problem, x)
            else:
                x = self.seeking_mode__(agent)
            child.solution = x
            child.flag = (
                True if self.generator.uniform() < self.mixture_ratio else False
            )
            n_population.append(child)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        self.population = self.population.spawn(self.population.evaluate(n_population, self.mode))
