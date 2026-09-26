#!/usr/bin/env python
# Created by "Thieu" at 09:16, 15/08/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


cdef class DevEPC(LegacyNativeOptimizer):
    """
    The developed version of: Emperor Penguins Colony (EPC)

    Notes:
        + This algorithm is almost like a trash algorithm. Some comments are as follows:
        + The code is incorrect and incomplete. It updates coefficients either increasing or decreasing, but the paper does not clearly provide any formulas describing how these increases or decreases are calculated.
        + Most of the formulas are wrong and meaningless, with no clear explanation of what the symbols represent. In particular, formulas 12 to 18 are problematic. There is no connection between the position update process in the algorithm and the parameters.
        + This algorithm can only be applied to 2-dimensional problems and cannot be extended to problems with more than 2 dimensions. The entire experimental section of the paper is also limited to 2-dimensional functions.
        + In the code, I simplified the position update process for penguins and modified the algorithm to work on n-dimensional problems. The parameter update rules were also devised by me. Therefore, I named it DevEPC.

    Links:
        1. https://doi.org/10.1007/s12065-019-00212-x

    Examples
    ~~~~~~~~
    >>> from clypto.collection.swarm_based import EPC    >>> import numpy as np
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
    >>> model = EPC.DevEPC(epoch=1000, pop_size=50, heat_damping_factor=0.95, mutation_factor=0.1,
    >>>                     spiral_a=1.0, spiral_b=0.5)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Harifi, S., Khalilian, M., Mohammadzadeh, J. and Ebrahimnejad, S., 2019.
    Emperor Penguins Colony: a new metaheuristic algorithm for optimization. Evolutionary intelligence, 12(2), pp.211-226.
    """

    cdef public object heat_damping_factor
    cdef public object mutation_factor
    cdef public object spiral_a
    cdef public object spiral_b
    cdef public object surface_area
    cdef public object emissivity
    cdef public object stefan_boltzmann
    cdef public object body_temperature
    cdef public object mu
    cdef public object heat_radiation
    cdef public object current_mutation_factor

    def __init__(
        self,
        epoch = 10000,
        pop_size = 100,
        heat_damping_factor: float = 0.95,
        mutation_factor: float = 0.5,
        spiral_a: float = 1.0,
        spiral_b: float = 0.5,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            heat_damping_factor (float): Damping factor for heat radiation, default = 0.95
            mutation_factor (float): Mutation factor for random movement, default = 0.1
            spiral_a (float): Constant for logarithmic spiral movement, default = 1.0
            spiral_b (float): Constant for logarithmic spiral movement, default = 0.5
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "heat_damping_factor",
                "mutation_factor",
                "spiral_a",
                "spiral_b",
            ],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.heat_damping_factor = cy.validator(float, heat_damping_factor, [0.0, 1.0], "heat_damping_factor")
        self.mutation_factor = cy.validator(float, mutation_factor, [0.0, 1.0], "mutation_factor")
        self.spiral_a = cy.validator(float, spiral_a, [0.0, 100.0], "spiral_a")
        self.spiral_b = cy.validator(float, spiral_b, [0.0, 100.0], "spiral_b")

    cdef void initialize_variables(self):
        # Physical constants (from paper)
        self.surface_area = 0.56  # m^2 (total surface area of emperor penguin)
        self.emissivity = 0.98  # emissivity of bird plumage
        self.stefan_boltzmann = 5.6703e-8  # W/m^2K^4
        self.body_temperature = 308.15  # K (35°C)
        self.mu = 0.01  # Attenuation coefficient (can be tuned)
        # Calculate heat radiation using Stefan-Boltzmann law (Equation 6)
        self.heat_radiation = (
                self.surface_area
                * self.emissivity
                * self.stefan_boltzmann
                * (self.body_temperature ** 4)
        )

    def calculate_attractiveness(self, heat_radiation: float, distance: float) -> float:
        # Linear heat source model with photon attenuation (Equations 9-11)
        if distance == 0:
            return heat_radiation
        else:
            # Heat intensity with linear source and attenuation
            return heat_radiation * np.exp(-self.mu * distance) / distance

    def spiral_movement(
            self, penguin_i: np.ndarray, penguin_j: np.ndarray, attractiveness: float
    ) -> np.ndarray:
        # Convert to polar coordinates
        diff = penguin_j - penguin_i
        # Simplified spiral movement calculation
        # Instead of complex polar coordinate transformation, use direct approach
        distance = np.linalg.norm(diff)
        if distance == 0 or np.allclose(penguin_i, penguin_j):
            return penguin_i.copy()
        # Direction vector
        direction = diff / distance
        # Spiral movement distance based on attractiveness
        move_distance = attractiveness * distance * self.spiral_a
        # Add spiral rotation effect
        theta = self.spiral_b * np.pi
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        # Apply rotation to direction (for 2D, extend for higher dimensions)
        if self.problem.n_dims >= 2:
            rotated_dir = direction.copy()
            rotated_dir[:2] = rotation_matrix @ direction[:2]
        else:
            rotated_dir = direction
        # Calculate new position - Add random component (mutation) - Equation 19
        new_position = (
                penguin_i
                + move_distance * rotated_dir
                + self.current_mutation_factor
                * self.generator.uniform(-1, 1, self.problem.n_dims)
        )
        return new_position

    cdef void evolve(self, int epoch_c):
        cdef NativePopulation pop = self.pop
        cdef NativePopulation cand
        cdef Py_ssize_t n = pop.n, d = pop.d
        cdef object rng = self.generator
        X = np.array(pop.X)
        F = np.asarray(pop.F)
        self.heat_radiation = self.heat_radiation * self.heat_damping_factor
        self.current_mutation_factor = self.mutation_factor * (1 - epoch_c / self.epoch)
        # every penguin moves towards each penguin that is better than it (one candidate per such pair)
        better = (F[None, :] < F[:, None]) if self.problem.minmax == "min" else (F[None, :] > F[:, None])
        i, j = np.nonzero(better)
        m = len(i)
        if m == 0:
            return
        diff = X[j] - X[i]
        dist = np.linalg.norm(diff, axis=1)
        att = np.where(dist == 0, self.heat_radiation, self.heat_radiation * np.exp(-self.mu * dist) / np.where(dist == 0, 1.0, dist))
        att = np.where(att > 1, 1.0 / (1.0 + att), att)
        direction = diff / np.where(dist == 0, 1.0, dist)[:, None]
        if d >= 2:
            theta = self.spiral_b * np.pi
            c, s = np.cos(theta), np.sin(theta)
            rotated = direction.copy()
            rotated[:, 0] = c * direction[:, 0] - s * direction[:, 1]
            rotated[:, 1] = s * direction[:, 0] + c * direction[:, 1]
            direction = rotated
        pos = X[i] + (att * dist * self.spiral_a)[:, None] * direction + self.current_mutation_factor * rng.uniform(-1, 1, (m, d))
        pos = np.where((dist == 0)[:, None], X[i], pos)
        cand = pop.take(i)
        cand.X[:] = self.correct_solution(pos)
        self.evaluate(cand, 0, m)
        ops.scatter(self, cand, i)
