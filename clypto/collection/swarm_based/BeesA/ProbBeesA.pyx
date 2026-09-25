#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class ProbBeesA(AgentListOptimizer):
    """
    The original version of: Probabilistic Bees Algorithm (P-BeesA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + recruited_bee_ratio (float): percent of bees recruited, default = 0.1
        + dance_factor (tuple, list): (radius, reduction) - Bees Dance Radius, default=(0.1, 0.99)

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import BeesA    >>> import numpy as np
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
    >>> model = BeesA.ProbBeesA(epoch=1000, pop_size=50, recruited_bee_ratio = 0.1, dance_radius = 0.1, dance_reduction = 0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T. and Castellani, M., 2015. A comparative study of the Bees Algorithm as a tool for
    function optimisation. Cogent Engineering, 2(1), p.1091540.
    """

    cdef public object recruited_bee_ratio
    cdef public object dance_radius
    cdef public object dance_reduction
    cdef public object dyn_radius
    cdef public object recruited_bee_count

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        recruited_bee_ratio: float = 0.1,
        dance_radius: float = 0.1,
        dance_reduction: float = 0.99,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float): percent of bees recruited, default = 0.1
            dance_radius (float): Bees Dance Radius, default=0.1
            dance_reduction (float): Bees Dance Radius Reduction Rate, default=0.99
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "recruited_bee_ratio",
                "dance_radius",
                "dance_reduction",
            ],
            sort_flag=True,
            parallelizable=True,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.recruited_bee_ratio = cy.validator(float, recruited_bee_ratio, (0, 1.0), "recruited_bee_ratio")
        self.dance_radius = cy.validator(float, dance_radius, (0, 1.0), "dance_radius")
        self.dance_reduction = cy.validator(float, dance_reduction, (0, 1.0), "dance_reduction")
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(list(range(0, self.problem.n_dims)))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return self.correct_solution(position)

    def evolve_agents(self, epoch):
        # Calculate Scores
        fit_list = np.array([agent.target.fitness for agent in self.objs])
        fit_list = 1.0 / (fit_list + self.EPSILON)
        d_fit = fit_list / np.mean(fit_list)
        for idx in range(0, self.pop_size):
            # Determine Rejection Probability based on Score
            if d_fit[idx] < 0.9:
                reject_prob = 0.6
            elif 0.9 <= d_fit[idx] < 0.95:
                reject_prob = 0.2
            elif 0.95 <= d_fit[idx] < 1.15:
                reject_prob = 0.05
            else:
                reject_prob = 0
            # Check for Acceptance/Rejection
            if self.generator.random() >= reject_prob:  # Acceptance
                # Calculate New Bees Count
                bee_count = int(np.ceil(d_fit[idx] * self.recruited_bee_count))
                if bee_count < 2:
                    bee_count = 2
                if bee_count > self.pop_size:
                    bee_count = self.pop_size
                # Create New Bees(Solutions)
                pop_child = []
                for j in range(0, bee_count):
                    pos_new = self.perform_dance__(
                        self.objs[idx].solution, self.dyn_radius
                    )
                    agent = self.generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
                pop_child = self.update_target_for_population(pop_child)
                local_best = self.get_best_agent(pop_child, self.problem.minmax)
                if self.compare_target(
                        local_best.target, self.objs[idx].target, self.problem.minmax
                ):
                    self.objs[idx] = local_best
            else:
                self.objs[idx] = self.generate_agent()
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius
