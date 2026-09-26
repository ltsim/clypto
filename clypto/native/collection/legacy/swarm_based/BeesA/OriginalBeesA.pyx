#!/usr/bin/env python
# Created by "Thieu" at 15:34, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class OriginalBeesA(LegacyOptimizer):
    """
    The original version of: Bees Algorithm (BeesA)

    Links:
        1. https://www.sciencedirect.com/science/article/pii/B978008045157250081X
        2. https://www.tandfonline.com/doi/full/10.1080/23311916.2015.1091540

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + selected_site_ratio (float): default = 0.5
        + elite_site_ratio (float): default = 0.4
        + selected_site_bee_ratio (float): default = 0.1
        + elite_site_bee_ratio (float): default = 2.0
        + dance_radius (float): default = 0.1
        + dance_reduction (float): default = 0.99

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import BeesA    >>> import numpy as np
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
    >>> model = BeesA.OriginalBeesA(epoch=1000, pop_size=50, selected_site_ratio=0.5, elite_site_ratio=0.4,
    >>>         selected_site_bee_ratio=0.1, elite_site_bee_ratio=2.0, dance_radius=0.1, dance_reduction=0.99)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Pham, D.T., Ghanbarzadeh, A., Koç, E., Otri, S., Rahim, S. and Zaidi, M., 2006. The bees algorithm—a novel tool
    for complex optimisation problems. In Intelligent production machines and systems (pp. 454-459). Elsevier Science Ltd.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            selected_site_ratio: float = 0.5,
            elite_site_ratio: float = 0.4,
            selected_site_bee_ratio: float = 0.1,
            elite_site_bee_ratio: float = 2.0,
            dance_radius: float = 0.1,
            dance_reduction: float = 0.99,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            selected_site_ratio (float):
            elite_site_ratio (float):
            selected_site_bee_ratio (float):
            elite_site_bee_ratio (float):
            dance_radius (float):
            dance_reduction (float):
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        # (Scout Bee Count or Population Size, Selected Sites Count)
        self.selected_site_ratio = self.validator.check_float(
            "selected_site_ratio", selected_site_ratio, (0, 1.0)
        )
        self.elite_site_ratio = self.validator.check_float(
            "elite_site_ratio", elite_site_ratio, (0, 1.0)
        )
        self.selected_site_bee_ratio = self.validator.check_float(
            "selected_site_bee_ratio", selected_site_bee_ratio, (0, 1.0)
        )
        self.elite_site_bee_ratio = self.validator.check_float(
            "elite_site_bee_ratio", elite_site_bee_ratio, (0, 3.0)
        )
        self.dance_radius = self.validator.check_float(
            "dance_radius", dance_radius, (0, 1.0)
        )
        self.dance_reduction = self.validator.check_float(
            "dance_reduction", dance_reduction, (0, 1.0)
        )
        self._set_parameters(
            [
                "epoch",
                "pop_size",
                "selected_site_ratio",
                "elite_site_ratio",
                "selected_site_bee_ratio",
                "elite_site_bee_ratio",
                "dance_radius",
                "dance_reduction",
            ]
        )
        # Initial Value of Dance Radius
        self.dyn_radius = self.dance_radius
        self.n_selected_bees = int(round(self.selected_site_ratio * self.pop_size))
        self.n_elite_bees = int(round(self.elite_site_ratio * self.n_selected_bees))
        self.n_selected_bees_local = int(
            round(self.selected_site_bee_ratio * self.pop_size)
        )
        self.n_elite_bees_local = int(
            round(self.elite_site_bee_ratio * self.n_selected_bees_local)
        )
        self.sort_flag = True

    def perform_dance__(self, position, r):
        jdx = self.generator.choice(range(0, self.problem.n_dims))
        position[jdx] = position[jdx] + r * self.generator.uniform(-1, 1)
        return self._correct_solution(position)

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = self.pop.copy()
        for idx in range(0, self.pop_size):
            # Elite Sites
            if idx < self.n_elite_bees:
                pop_child = []
                for j in range(0, self.n_elite_bees_local):
                    pos_new = self.perform_dance__(
                        self.pop[idx].solution, self.dyn_radius
                    )
                    agent = self._generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self._get_target(pos_new)
                pop_child = self._update_target_for_population(pop_child)
                local_best = self._get_best_agent(pop_child, self.problem.sense)
                if self._compare_target(
                        local_best.target, self.pop[idx].target, self.problem.sense
                ):
                    pop_new[idx] = local_best
            elif self.n_elite_bees <= idx < self.n_selected_bees:
                # Selected Non-Elite Sites
                pop_child = []
                for j in range(0, self.n_selected_bees_local):
                    pos_new = self.perform_dance__(
                        self.pop[idx].solution, self.dyn_radius
                    )
                    agent = self._generate_empty_agent(pos_new)
                    pop_child.append(agent)
                    if self.mode not in self.AVAILABLE_MODES:
                        pop_child[-1].target = self._get_target(pos_new)
                pop_child = self._update_target_for_population(pop_child)
                local_best = self._get_best_agent(pop_child, self.problem.sense)
                if self._compare_target(
                        local_best.target, self.pop[idx].target, self.problem.sense
                ):
                    pop_new[idx] = local_best
            else:
                # Non-Selected Sites
                pop_new[idx] = self._generate_agent()
        self.pop = pop_new
        # Damp Dance Radius
        self.dyn_radius = self.dance_reduction * self.dance_radius
