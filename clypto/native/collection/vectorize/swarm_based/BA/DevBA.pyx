#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 12:00, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class DevBA(AgentListOptimizer):
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
    >>> from clypto.collection.swarm_based import BA    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> model = BA.DevBA(epoch=1000, pop_size=50, pulse_rate = 0.95, pf_min = 0., pf_max = 10.)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")
    """

    cdef public object pulse_rate
    cdef public object pf_min
    cdef public object pf_max
    cdef public object alpha
    cdef public object gamma
    cdef public object dyn_list_velocity

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        pulse_rate = 0.95,
        pf_min = 0.0,
        pf_max = 10.0,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "pulse_rate", "pf_min", "pf_max"],
            sort_flag=False,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.pulse_rate = cy.validator(float, pulse_rate, (0, 1.0), "pulse_rate")
        self.pf_min = cy.validator(float, pf_min, [0, 2], "pf_min")
        self.pf_max = cy.validator(float, pf_max, [2, 10], "pf_max")
        self.alpha = self.gamma = 0.9

    cdef void initialize_variables(self):
        self.dyn_list_velocity = np.zeros((self.pop_size, self.problem.n_dims))

    def evolve_agents(self, epoch):
        pop_new = []
        for idx in range(0, self.pop_size):
            pf = (
                self.pf_min + (self.pf_max - self.pf_min) * self.generator.uniform()
            )  # Eq. 2
            self.dyn_list_velocity[idx] = (
                self.generator.uniform() * self.dyn_list_velocity[idx]
                + (self.g_best.solution - self.objs[idx].solution) * pf
            )  # Eq. 3
            x = self.objs[idx].solution + self.dyn_list_velocity[idx]  # Eq. 4
            pos_new = self.correct_solution(x)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        pop_child_idx = []
        pop_child = []
        for idx in range(0, self.pop_size):
            if self.compare_target(
                pop_new[idx].target, self.objs[idx].target, self.problem.minmax
            ):
                self.objs[idx].update(
                    solution=pop_new[idx].solution.copy(), target=pop_new[idx].target
                )
            else:
                if self.generator.random() > self.pulse_rate:
                    x = self.g_best.solution + 0.01 * self.generator.uniform(
                        self.problem.lb, self.problem.ub
                    )
                    pos_new = self.correct_solution(x)
                    agent = self.generate_empty_agent(pos_new)
                    pop_child_idx.append(idx)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
        pop_child = self.update_target_for_population(pop_child)
        for idx, idx_selected in enumerate(pop_child_idx):
            if self.compare_target(
                pop_child[idx].target, pop_new[idx_selected].target, self.problem.minmax
            ):
                pop_new[idx_selected].update(
                    solution=pop_child[idx].solution, target=pop_child[idx].target
                )
        self.objs = pop_new
