#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.optimizer.native.legacy cimport _LegacyOptimizer


cdef class ProbBeesA(_LegacyOptimizer):
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

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            recruited_bee_ratio: float = 0.1,
            dance_radius: float = 0.1,
            dance_reduction: float = 0.99,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            recruited_bee_ratio (float): percent of bees recruited, default = 0.1
            dance_radius (float): Bees Dance Radius, default=0.1
            dance_reduction (float): Bees Dance Radius Reduction Rate, default=0.99
        """
        _LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.recruited_bee_ratio = self.validator.check_float(
            "recruited_bee_ratio", recruited_bee_ratio, (0, 1.0)
        )
        self.dance_radius = self.validator.check_float(
            "dance_radius", dance_radius, (0, 1.0)
        )
        self.dance_reduction = self.validator.check_float(
            "dance_reduction", dance_reduction, (0, 1.0)
        )
        self.set_parameters(
            [
                "epoch",
                "pop_size",
                "recruited_bee_ratio",
                "dance_radius",
                "dance_reduction",
            ]
        )
        self.sort_flag = True
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.recruited_bee_count = int(round(self.recruited_bee_ratio * self.pop_size))

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(list(range(0, self.problem.n_dims)))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return self.correct_solution(position)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        # Calculate Scores
        fit_list = np.array([agent.target.fitness for agent in self.pop])
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
                        self.pop[idx].solution, self.dyn_radius
                    )
                    agent = self.generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self.get_target(pos_new)
                pop_child = self.update_target_for_population(pop_child)
                local_best = self.get_best_agent(pop_child, self.problem.minmax)
                if self.compare_target(
                        local_best.target, self.pop[idx].target, self.problem.minmax
                ):
                    self.pop[idx] = local_best
            else:
                self.pop[idx] = self.generate_agent()
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius
