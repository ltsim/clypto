#!/usr/bin/env python
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import numpy as np
cimport clypto.core as cy



cdef class DevBA(cy.Optimizer):
    """
    The original version of: Developed Bat-inspired Algorithm (DBA)

    Notes
    ~~~~~
    + A (loudness) parameter is removed
    + Flow is changed:
        + 1st: the exploration phase is proceed (using frequency)
        + 2nd: If new position has better fitness, replace the old position
        + 3rd: Otherwise, proceed exploitation phase (using finding around the best position so far)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + pulse_rate (float): [0.7, 1.0], pulse rate / emission rate, default = 0.95
        + pulse_frequency (tuple, list): (pf_min, pf_max) -> ([0, 3], [5, 20]), pulse frequency, default = (0, 10)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BA    >>> import numpy as np
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
    >>> model = BA.DevBA(epoch=1000, pop_size=50, pulse_rate = 0.95, pf_min = 0., pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")
    """

    cdef public double pf_max
    cdef public double pf_min
    cdef public double pulse_rate

    def __init__(
        self,
        epoch=10000,
        pop_size=100,
        pulse_rate=0.95,
        pf_min=0.0,
        pf_max=10.0,
        **kwargs
    ):
        super().__init__(parameters=["epoch", "pop_size", "pulse_rate", "pf_min", "pf_max"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size)
        self.pulse_rate = cy.validator(float, pulse_rate, (0, 1.0), "pulse_rate")
        self.pf_min = cy.validator(float, pf_min, [0, 2], "pf_min")
        self.pf_max = cy.validator(float, pf_max, [2, 10], "pf_max")
        self.alpha = self.gamma = 0.9

    def initialize_variables(self):
        pop_size = self.population.size()
        self.dyn_list_velocity = np.zeros((pop_size, self.problem.n_dims))

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        pop_new = []
        for idx in range(0, pop_size):
            pf = (
                self.pf_min + (self.pf_max - self.pf_min) * self.generator.uniform()
            )  # Eq. 2
            self.dyn_list_velocity[idx] = (
                self.generator.uniform() * self.dyn_list_velocity[idx]
                + (self.g_best.solution - self.population[idx].solution) * pf
            )  # Eq. 3
            x = self.population[idx].solution + self.dyn_list_velocity[idx]  # Eq. 4
            pos_new = self.population.correct_solution(x)
            agent = self.population.create_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].evaluate(self.problem)
        pop_new = self.population.evaluate(pop_new, self.mode)
        pop_child_idx = []
        pop_child = []
        for idx in range(0, pop_size):
            if cy.is_better(pop_new[idx], self.population[idx], self.problem.sense):
                self.population[idx].update_solution(pop_new[idx], pop_new[idx].solution.copy())
            else:
                if self.generator.random() > self.pulse_rate:
                    x = self.g_best.solution + 0.01 * self.generator.uniform(
                        self.problem.bounds.low, self.problem.bounds.up
                    )
                    pos_new = self.population.correct_solution(x)
                    agent = self.population.create_agent(pos_new)
                    pop_child_idx.append(idx)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].evaluate(self.problem)
        pop_child = self.population.evaluate(pop_child, self.mode)
        for idx, idx_selected in enumerate(pop_child_idx):
            if cy.is_better(pop_child[idx], pop_new[idx_selected], self.problem.sense):
                pop_new[idx_selected].update_solution(pop_child[idx], pop_child[idx].solution)
        self.population = pop_new
