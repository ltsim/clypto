#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class AdaptiveBAAgent(cy.Agent):
    cdef public object velocity
    cdef public object loudness
    cdef public object pulse_rate

    def __init__(self, solution=None, objectives=None, weights=None, velocity=None, loudness=None, pulse_rate=None):
        cy.Agent.__init__(self, solution, objectives, weights)
        self.velocity = velocity
        self.loudness = loudness
        self.pulse_rate = pulse_rate

    cdef cy.Agent clone(self):
        cdef AdaptiveBAAgent new = <AdaptiveBAAgent>cy.Agent.clone(self)
        new.velocity = self.velocity
        new.loudness = self.loudness
        new.pulse_rate = self.pulse_rate
        return new


cdef class AdaptiveBAPopulation(cy.Population):
    """Agents of :class:`AdaptiveBA`."""
    cdef public object loudness_max
    cdef public object loudness_min
    cdef public object pr_max
    cdef public object pr_min

    cdef void copy_state(self, cy.Population new):
        cy.Population.copy_state(self, new)
        (<AdaptiveBAPopulation>new).loudness_max = self.loudness_max
        (<AdaptiveBAPopulation>new).loudness_min = self.loudness_min
        (<AdaptiveBAPopulation>new).pr_max = self.pr_max
        (<AdaptiveBAPopulation>new).pr_min = self.pr_min

    def create_agent(self, solution: np.ndarray | None = None) -> cy.Agent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(self.problem.bounds.low, self.problem.bounds.up)
        loudness = self.generator.uniform(self.loudness_min, self.loudness_max)
        pulse_rate = self.generator.uniform(self.pr_min, self.pr_max)
        return AdaptiveBAAgent(
            solution=solution,
            velocity=velocity,
            loudness=loudness,
            pulse_rate=pulse_rate,
        )


cdef class AdaptiveBA(cy.Optimizer):
    """
    The original version of: Adaptive Bat-inspired Algorithm (ABA)

    Notes
    ~~~~~
    + The value of A and r are changing after each iteration

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + loudness_min (float): A_min - loudness, default=1.0
        + loudness_max (float): A_max - loudness, default=2.0
        + pr_min (float): pulse rate / emission rate min, default = 0.15
        + pr_max (float): pulse rate / emission rate max, default = 0.85
        + pf_min (float): pulse frequency min, default = 0
        + pf_max (float): pulse frequency max, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.swarm_based import BA    >>> import numpy as np
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
    >>> model = BA.AdaptiveBA(epoch=1000, pop_size=50, loudness_min = 1.0, loudness_max = 2.0, pr_min = -2.5, pr_max = 0.85, pf_min = 0.1, pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Yang, X.S., 2010. A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative
    strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.
    """

    cdef public double loudness_max
    cdef public double loudness_min
    cdef public double pf_max
    cdef public double pf_min
    cdef public double pr_max
    cdef public double pr_min

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: object = 100,
        loudness_min: float = 1.0,
        loudness_max: float = 2.0,
        pr_min: float = 0.15,
        pr_max: float = 0.85,
        pf_min: float = -10.0,
        pf_max: float = 10.0,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            loudness_min (float): A_min - loudness, default=1.0
            loudness_max (float): A_max - loudness, default=2.0
            pr_min (float): pulse rate / emission rate min, default = 0.15
            pr_max (float): pulse rate / emission rate max, default = 0.85
            pf_min (float): pulse frequency min, default = 0
            pf_max (float): pulse frequency max, default = 10
        """
        super().__init__(parameters=[ "epoch", "pop_size", "loudness_min", "loudness_max", "pr_min", "pr_max", "pf_min", "pf_max", ], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[5, 10000], cls=AdaptiveBAPopulation)
        self.loudness_min = cy.validator(float, loudness_min, [0.5, 1.5], "loudness_min")
        self.loudness_max = cy.validator(float, loudness_max, [1.5, 3.0], "loudness_max")
        self.population.loudness_min = self.loudness_min
        self.pr_min = cy.validator(float, pr_min, [-10.0, 10.0], "pr_min")
        self.population.pr_min = self.pr_min
        self.population.loudness_max = self.loudness_max
        self.pr_max = cy.validator(float, pr_max, [-10.0, 10.0], "pr_max")
        self.pf_min = cy.validator(float, pf_min, [-10.0, 10.0], "pf_min")
        self.population.pr_max = self.pr_max
        self.pf_max = cy.validator(float, pf_max, [0.0, 10.0], "pf_max")
        self.alpha = self.gamma = 0.9

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        mean_a = np.mean([agent.loudness for agent in self.population])
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx in range(0, pop_size):
            agent = cy.duplicate_agent(self.population[idx])
            pulse_frequency = self.generator.uniform(self.pf_min, self.pf_max)
            agent.velocity = agent.velocity + pulse_frequency * (
                self.population[idx].solution - self.g_best.solution
            )
            x_new = self.population[idx].solution + agent.velocity
            ## Local Search around g_best position
            if self.generator.random() > agent.pulse_rate:
                x_new = self.g_best.solution + mean_a * self.generator.normal(-1, 1)
            x = cy.correct_solution(self.problem, x_new)
            agent.solution = x
            n_population.append(agent)
            if self.mode == "sequential":
                n_population[-1].evaluate(self.problem)
        n_population = self.population.evaluate(n_population, self.mode)
        for idx, agent in enumerate(self.population.toarray()):
            ## Replace the old position by the new one when its has better fitness.
            ##  and then update loudness and emission rate
            if (
                cy.is_better(n_population[idx], agent, self.problem.sense)
                and self.generator.random() < n_population[idx].loudness
            ):
                loudness = self.alpha * n_population[idx].loudness
                pulse_rate = n_population[idx].pulse_rate * (1 - np.exp(-self.gamma * epoch))
                agent.update_solution(n_population[idx], n_population[idx].solution)
                agent.loudness = loudness
                agent.pulse_rate = pulse_rate
