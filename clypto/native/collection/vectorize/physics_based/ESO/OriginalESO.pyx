#!/usr/bin/env python
# Created by "Thieu" at 07:03, 16/07/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.target cimport NativeTarget


cdef class OriginalESO(LegacyNativeOptimizer):
    """
    The original version of: Electrical Storm Optimization (ESO)

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import ESO    >>> import numpy as np
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
    >>> model = ESO.OriginalESO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Soto Calvo, Manuel, and Han Soo Lee. 2025. "Electrical Storm Optimization (ESO) Algorithm: Theoretical Foundations, Analysis, and Application to Engineering Problems" Machine Learning and Knowledge Extraction 7, no. 1: 24. https://doi.org/10.3390/make7010024
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation out = pop.empty_like()
        cdef NativeTarget tar_a, tar_b
        cdef Py_ssize_t idx
        ## Calculate storm parameters
        # Calculate field resistance based on population spread
        pos_pop = np.array(pop.X)
        mean_pos = np.mean(pos_pop, axis=0)
        std_pos = np.sqrt(np.mean(np.sum((pos_pop - mean_pos) ** 2, axis=1)))
        # std_pos = np.std(pos_pop, axis=0)
        peak_to_peak = np.max(np.max(pos_pop, axis=0) - np.min(pos_pop, axis=0))
        # Field resistance
        if peak_to_peak <= 0:
            resistance = 0
            ionized_pop = np.array([], dtype=int)
        else:
            resistance = std_pos / peak_to_peak
            # Identify ionized areas (promising regions)
            # Calculate percentile threshold
            percentile_threshold = (resistance / 2) * 100
            # Find solutions better than percentile
            fits = np.array(pop.F)
            fitness_percentile = np.percentile(fits, percentile_threshold)
            ionized_indices = np.where(fits <= fitness_percentile)[0]
            ionized_pop = ionized_indices

        # Calculate field conductivity using logistic function
        if resistance <= 0:
            fc = 1.0
        else:
            # Beta calculation (logistic function)
            try:
                exp_term = np.exp(resistance) / resistance
                log_term = np.log(1.0 - resistance) if resistance < 1 else 0
                beta = 1.0 / (1.0 + np.exp(-exp_term) * (resistance - abs(log_term)))
            except (OverflowError, ValueError):
                beta = 0.5
            try:
                fc = (
                        np.exp(resistance)
                        + np.exp(1 - resistance) * abs(np.log(resistance)) * beta
                )
            except (OverflowError, ValueError):
                fc = 1.0

        # Calculate field intensity using logistic function
        if resistance <= 0:
            fi = fc
        else:
            # Gamma calculation
            try:
                exp_term = np.exp(resistance) / resistance
                iter_ratio = epoch / self.epoch
                log_term = np.log(1 - iter_ratio) if iter_ratio < 1 else 0
                gama = 1 / (1 + np.exp(-exp_term * (resistance - abs(log_term))))
            except (OverflowError, ValueError):
                gama = 0.5
            fi = fc * gama

        # Calculate storm power
        if fc > 0:
            storm_power = (resistance * fi) / fc
        else:
            storm_power = 0

        # Update each lighting agent
        for idx in range(0, self.pop_size):
            # Initialize new lighting position
            if idx == 0 or len(ionized_pop) == 0:
                pos_a = self.problem.generate_solution(True)
            else:
                # Initialize near ionized areas
                alpha = ionized_pop[self.generator.integers(0, len(ionized_pop))]
                perturbation = self.generator.normal(
                    loc=0, scale=storm_power, size=self.problem.n_dims
                )
                pos_new = pop.X[alpha] + perturbation
                pos_a = self.correct_solution(pos_new)
                self.get_target(pos_a)  # generate_agent evaluates the position ...
            tar_a = self.get_target(pos_a)  # ... and the classic code evaluates it once more

            ## Branching and propagation
            # Simulate branching and propagation of lightning
            in_ionized = False
            for alpha in ionized_pop:
                if np.linalg.norm(pos_a - pop.X[alpha]) < 0.1:
                    in_ionized = True
                    break

            if in_ionized:
                # Propagate within ionized area
                pos_new = pos_a * storm_power
            else:
                # Propagate towards ionized areas
                if len(ionized_pop) > 0:
                    # Average position of ionized areas
                    avg_ionized = np.mean(np.array(pop.X[ionized_pop]), axis=0)
                    # Random perturbation
                    pos_new = avg_ionized + storm_power * np.exp(
                        fc
                    ) * self.generator.uniform(-fc, fc, self.problem.n_dims)
                else:
                    # Random search
                    pos_new = self.generator.uniform(
                        self.problem.lb, self.problem.ub, self.problem.n_dims
                    )
            pos_b = self.correct_solution(pos_new)
            tar_b = self.get_target(pos_b)

            # Select better position (the classic code compares with the default minmax="min")
            if tar_b.fitness < tar_a.fitness:
                ops.set_row(out, idx, pos_b, tar_b)
            else:
                ops.set_row(out, idx, pos_a, tar_a)
        self.pop = out
