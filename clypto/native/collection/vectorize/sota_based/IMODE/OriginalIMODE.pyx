#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 18:36, 16/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import Tuple, List
import numpy as np
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.agent_list cimport AgentListOptimizer
from clypto.optimizer._native.agent_list import FieldAgent


cdef class OriginalIMODE(AgentListOptimizer):
    """
    The original version of: Improved Multi-operator Differential Evolution Algorithm (IMODE)

    Links:
        1. https://doi.org/10.1109/CEC48606.2020.9185577
        2. This version is conversion from the original MATLAB code available at CEC competition github.

    Examples
    ~~~~~~~~
    >>> from clypto.collection.sota_based import IMODE    >>> import numpy as np
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
    >>> model = IMODE.OriginalIMODE(epoch=1000, pop_size=50, memory_size=5, archive_size=20)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Sallam, K. M., Elsayed, S. M., Chakrabortty, R. K., & Ryan, M. J. (2020, July). Improved multi-operator
    differential evolution algorithm for solving unconstrained problems. In 2020 IEEE congress on
    evolutionary computation (CEC) (pp. 1-8). IEEE.
    """

    cdef public object memory_size
    cdef public object archive_size
    cdef public object operator_probs
    cdef public object memory_pos
    cdef public object memory_f
    cdef public object memory_cr
    cdef public object archive

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        memory_size: int = 5,
        archive_size: int = 20,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            memory_size (int): [2, 20], Memory size for F and CR, default = 5
            archive_size (int): [5, 100], Size of the solution archive for diversity, default = 20
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "memory_size", "archive_size"],
            sort_flag=True,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.memory_size = cy.validator(int, memory_size, [2, 100], "memory_size")
        self.archive_size = cy.validator(int, archive_size, [5, 100], "archive_size")

    cdef void initialize_variables(self):
        # Operator probabilities (3 operators)
        self.operator_probs = np.ones(3) / 3
        # Parameter memory for adaptive control
        self.memory_pos = 0
        self.memory_f = np.full(self.memory_size, 0.5)  # Scaling factor memory
        self.memory_cr = np.full(self.memory_size, 0.5)  # Crossover rate memory
        # Archive for diversity
        self.archive_size = max(self.archive_size, self.pop_size)

    cdef void before_main_loop(self):
        # Initialize archive with initial population
        self.archive = self.objs.copy()

    def _generate_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        # Select random memory indices
        mem_indices = self.generator.integers(0, self.memory_size, self.pop_size)
        mu_f = self.memory_f[mem_indices]
        mu_cr = self.memory_cr[mem_indices]

        # Generate CR with normal distribution
        cr = self.generator.normal(mu_cr, 0.1)
        cr[mu_cr == -1] = 0  # Handle special case
        cr = np.clip(cr, 0, 1)

        # Generate F with Cauchy distribution
        f = mu_f + 0.1 * np.tan(np.pi * (self.generator.random(self.pop_size) - 0.5))

        # Regenerate negative F values
        negative_mask = f <= 0
        while np.any(negative_mask):
            f[negative_mask] = mu_f[negative_mask] + 0.1 * np.tan(
                np.pi * (self.generator.random(np.sum(negative_mask)) - 0.5)
            )
            negative_mask = f <= 0
        f = np.clip(f, 0, 1)
        return f, cr

    def _select_operator_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rand_vals = self.generator.random(self.pop_size)
        prob_cumsum = np.cumsum(self.operator_probs)

        op1_mask = rand_vals <= prob_cumsum[0]
        op2_mask = (rand_vals > prob_cumsum[0]) & (rand_vals <= prob_cumsum[1])
        op3_mask = rand_vals > prob_cumsum[1]

        return op1_mask, op2_mask, op3_mask

    def _generate_random_indices(
            self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        combined_pop = self.objs + self.archive
        total_size = len(combined_pop)

        # Generate unique random indices and Ensure indices are different
        r1, r2, r3 = (
            np.zeros(self.pop_size, dtype=np.intp),
            np.zeros(self.pop_size, dtype=np.intp),
            np.zeros(self.pop_size, dtype=np.intp),
        )
        for idx in range(0, self.pop_size):
            x1, x3 = self.generator.choice(
                list(set(range(self.pop_size)) - {idx}), size=2, replace=False
            )
            x2 = self.generator.choice(list(set(range(total_size)) - {idx, x1, x3}))
            r1[idx] = x1
            r2[idx] = x2
            r3[idx] = x3
        return r1, r2, r3, combined_pop

    def _mutation(self, f: np.ndarray) -> np.ndarray:
        # Select operators for each individual
        op1_mask, op2_mask, op3_mask = self._select_operator_indices()
        # Generate random indices
        r1, r2, r3, combined_pop = self._generate_random_indices()
        # Initialize mutant vectors
        matrix_pos = np.array([agent.solution for agent in self.objs])
        matrix_combined = np.array([agent.solution for agent in combined_pop])
        matrix_mutant = np.zeros_like(matrix_pos)

        # Operator 1: DE/current-to-pbest/1/bin-archive
        if np.any(op1_mask):
            p_best_size = max(int(0.25 * self.pop_size), 1)
            pbest_indices = self.generator.integers(0, p_best_size, self.pop_size)
            matrix_pbest = matrix_pos[pbest_indices]
            matrix_mutant[op1_mask] = matrix_pos[op1_mask] + f[op1_mask, np.newaxis] * (
                    matrix_pbest[op1_mask]
                    - matrix_pos[op1_mask]
                    + matrix_pos[r1[op1_mask]]
                    - matrix_combined[r2[op1_mask]]
            )
        # Operator 2: DE/current-to-pbest/1/bin
        if np.any(op2_mask):
            p_best_size = max(int(0.25 * self.pop_size), 1)
            pbest_indices = self.generator.integers(0, p_best_size, self.pop_size)
            matrix_pbest = matrix_pos[pbest_indices]
            matrix_mutant[op2_mask] = matrix_pos[op2_mask] + f[op2_mask, np.newaxis] * (
                    matrix_pbest[op2_mask]
                    - matrix_pos[op2_mask]
                    + matrix_pos[r1[op2_mask]]
                    - matrix_pos[r3[op2_mask]]
            )
        # Operator 3: DE/rand-to-pbest/1
        if np.any(op3_mask):
            p_best_size = max(int(0.5 * self.pop_size), 2)
            pbest_indices = self.generator.integers(0, p_best_size, self.pop_size)
            matrix_pbest = matrix_pos[pbest_indices]
            matrix_mutant[op3_mask] = f[op3_mask, np.newaxis] * matrix_pos[
                r1[op3_mask]
            ] + f[op3_mask, np.newaxis] * (
                                              matrix_pbest[op3_mask] - matrix_pos[r3[op3_mask]]
                                      )
        return matrix_mutant

    def _handle_boundaries(self, vectors: np.ndarray) -> np.ndarray:
        strategy = self.generator.integers(1, 4)
        result = []

        if strategy == 1:  # Strategy 1: Midpoint repair
            for idx in range(0, len(vectors)):
                res = np.select(
                    [vectors[idx] < self.problem.lb, vectors[idx] > self.problem.ub],
                    [
                        (vectors[idx] + self.problem.ub) / 2,
                        (vectors[idx] + self.problem.lb) / 2,
                    ],
                    default=vectors[idx],
                )
                result.append(res)
        elif strategy == 2:  # Strategy 2: Reflection
            for idx in range(0, len(vectors)):
                res = vectors[idx]
                flag1 = res < self.problem.lb
                res[flag1] = np.clip(
                    2 * self.problem.lb[flag1] - res[flag1],
                    self.problem.lb[flag1],
                    self.problem.ub[flag1],
                )
                flag2 = res > self.problem.ub
                res[flag2] = np.clip(
                    2 * self.problem.ub[flag2] - res[flag2],
                    self.problem.lb[flag2],
                    self.problem.ub[flag2],
                )
                result.append(res)
        else:  # Strategy 3: Random reinitialization
            for idx in range(0, len(vectors)):
                res = vectors[idx]
                mask_lower = res < self.problem.lb
                mask_upper = res > self.problem.ub
                res[mask_lower | mask_upper] = self.generator.uniform(
                    self.problem.lb[mask_lower | mask_upper],
                    self.problem.ub[mask_lower | mask_upper],
                )
                result.append(res)
        results = np.clip(
            result, self.problem.lb, self.problem.ub
        )  # Ensure final results are within bounds
        return results

    def _crossover(self, mutant: np.ndarray, cr: np.ndarray) -> np.ndarray:
        matrix_pos = np.array([agent.solution for agent in self.objs])
        if self.generator.random() < 0.4:
            # Binomial crossover
            cross_mask = (
                    self.generator.random((self.pop_size, self.problem.n_dims))
                    <= cr[:, np.newaxis]
            )
            # Ensure at least one dimension is taken from mutant
            for idx in range(self.pop_size):
                if not np.any(cross_mask[idx]):
                    cross_mask[idx, self.generator.integers(0, self.problem.n_dims)] = (
                        True
                    )
            trial = matrix_pos.copy()
            trial[cross_mask] = mutant[cross_mask]
        else:
            # Exponential crossover
            trial = matrix_pos.copy()
            start_points = self.generator.integers(
                0, self.problem.n_dims, self.pop_size
            )
            for idx in range(self.pop_size):
                jdx = start_points[idx]
                while self.generator.random() < cr[idx] and jdx < self.problem.n_dims:
                    trial[idx, jdx] = mutant[idx, jdx]
                    jdx += 1
        return trial

    def _update_archive(self, improved_pop=None):
        if len(improved_pop) == 0:
            return

        # Add new solutions to archive
        if len(self.archive) == 0:
            self.archive = improved_pop
        else:
            self.archive = self.archive + improved_pop

        # Remove duplicates and maintain size limit
        if len(self.archive) > 1:
            ## Remove duplicates based on position
            self.archive = list(set(self.archive))

            ## Randomly remove excess individuals from archive
            if len(self.archive) > self.archive_size:
                remove_count = len(self.archive) - self.archive_size
                remove_indices = self.generator.choice(
                    len(self.archive), remove_count, replace=False
                )
                keep_indices = list(set(range(len(self.archive))) - set(remove_indices))
                self.archive = [self.archive[idx] for idx in keep_indices]

    def evolve_agents(self, epoch):
        # Generate adaptive parameters
        f_values, cr_values = self._generate_parameters()

        # Sort population by fitness
        self.objs = self.get_sorted_population(self.objs, self.problem.minmax)
        cr_values = np.sort(cr_values)

        # Mutation
        matrix_mutant = self._mutation(f_values)

        # Handle boundaries
        matrix_mutant = self._handle_boundaries(matrix_mutant)

        # Crossover
        matrix_child = self._crossover(matrix_mutant, cr_values)

        # Evaluate trial population
        improvement_mask = np.zeros(self.pop_size, dtype=bool)
        improvements = np.zeros(self.pop_size)
        pop_new = []
        for idx in range(len(matrix_child)):
            pos_new = self.correct_solution(matrix_child[idx])
            agent = self.generate_agent(pos_new)
            if self.compare_target(agent.target, self.objs[idx].target):
                improvement_mask[idx] = True
            improvements[idx] = np.abs(
                self.objs[idx].target.fitness - agent.target.fitness
            )
            pop_new.append(agent)

        # Track operator performance
        op1_mask, op2_mask, op3_mask = self._select_operator_indices()
        fits = np.array([agent.target.fitness for agent in self.objs])
        fits_child = np.array([agent.target.fitness for agent in pop_new])
        relative_improvements = np.maximum(0, (fits - fits_child) / np.abs(fits))

        # Update archive with improved solutions
        if np.any(improvement_mask):
            self._update_archive(
                [pop_new[idx] for idx in range(self.pop_size) if improvement_mask[idx]]
            )

        # Update parameters
        if np.any(improvement_mask):
            successful_f = f_values[improvement_mask]
            successful_cr = cr_values[improvement_mask]
            successful_improvements = improvements[improvement_mask]
            ### Update parameter memory based on successful parameters
            if len(successful_f) > 0:
                # Weight by improvement amount
                weights = successful_improvements / np.sum(successful_improvements)
                # Update F memory (Lehmer mean)
                self.memory_f[self.memory_pos] = np.sum(
                    weights * successful_f ** 2
                ) / np.sum(weights * successful_f)
                # Update CR memory
                if np.max(successful_cr) == 0:
                    self.memory_cr[self.memory_pos] = -1
                else:
                    self.memory_cr[self.memory_pos] = np.sum(
                        weights * successful_cr ** 2
                    ) / np.sum(weights * successful_cr)
                # Update memory position
                self.memory_pos = (self.memory_pos + 1) % self.memory_size
            else:
                # No successful parameters, use default values
                self.memory_f[self.memory_pos] = 0.5
                self.memory_cr[self.memory_pos] = 0.5

        ## Update operator selection probabilities
        op1_improvement = (
            np.mean(relative_improvements[op1_mask]) if np.any(op1_mask) else 0
        )
        op2_improvement = (
            np.mean(relative_improvements[op2_mask]) if np.any(op2_mask) else 0
        )
        op3_improvement = (
            np.mean(relative_improvements[op3_mask]) if np.any(op3_mask) else 0
        )
        total_improvement = op1_improvement + op2_improvement + op3_improvement
        if total_improvement > 0:
            self.operator_probs = np.array(
                [op1_improvement, op2_improvement, op3_improvement]
            )
            self.operator_probs = np.clip(
                self.operator_probs / total_improvement, 0.1, 0.9
            )
            self.operator_probs = self.operator_probs / np.sum(
                self.operator_probs
            )  # Normalize
        else:
            self.operator_probs = np.ones(3) / 3

        ## Update population
        self.objs = [
            a if flag else b for a, b, flag in zip(pop_new, self.objs, improvement_mask)
        ]
