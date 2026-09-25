#!/usr/bin/env python
# Created by "Thieu" at 17:48, 18/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from clypto.native.collection.legacy.music_based.HS.DevHS cimport DevHS


cdef class OriginalHS(DevHS):
    """
    The original version of: Harmony Search (HS)

    Links:
        1. https://doi.org/10.1177/003754970107600201

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_r (float): [0.1, 0.5], Harmony Memory Consideration Rate), default = 0.15
        + pa_r (float): [0.3, 0.8], Pitch Adjustment Rate, default=0.5

    Examples
    ~~~~~~~~
    >>> from clypto.collection.music_based import HS    >>> import numpy as np
    >>> from clypto import FloatVar
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    >>>     "minmax": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = HS.OriginalHS(epoch=1000, pop_size=50, c_r = 0.95, pa_r = 0.05)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Geem, Z.W., Kim, J.H. and Loganathan, G.V., 2001. A new heuristic
    optimization algorithm: harmony search. simulation, 76(2), pp.60-68.
    """

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            c_r: float = 0.95,
            pa_r: float = 0.05,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            c_r (float): Harmony Memory Consideration Rate), default = 0.15
            pa_r (float): Pitch Adjustment Rate, default=0.5
        """
        super().__init__(epoch, pop_size, c_r, pa_r, **kwargs)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from _LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = self.generator.uniform(self.problem.lb, self.problem.ub)
            for jdx in range(self.problem.n_dims):
                # Use Harmony Memory
                if self.generator.uniform() <= self.c_r:
                    random_index = self.generator.integers(0, self.pop_size)
                    pos_new[jdx] = self.pop[random_index].solution[jdx]
                # Pitch Adjustment
                if self.generator.uniform() <= self.pa_r:
                    mean = (self.problem.lb + self.problem.ub) / 2
                    std_dev = (
                            abs(self.problem.ub - self.problem.lb) / 6
                    )  # This assumes a range of +/- 3 standard deviations
                    delta = self.dyn_fw * self.generator.normal(
                        mean, std_dev
                    )  # Gaussian(Normal)
                    pos_new[jdx] = pos_new[jdx] + delta[jdx]
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                pop_new[-1].target = self.get_target(pos_new)
        pop_new = self.update_target_for_population(pop_new)
        # Update Damp Fret Width
        self.dyn_fw = self.dyn_fw * self.fw_damp
        # Merge Harmony Memory and New Harmonies, Then sort them, Then truncate extra harmonies
        self.pop = self.get_sorted_and_trimmed_population(
            self.pop + pop_new, self.pop_size, minmax=self.problem.minmax
        )
