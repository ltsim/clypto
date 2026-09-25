#!/usr/bin/env python
# Created by "Thieu" at 21:19, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.collection.physics_based.EFO.DevEFO cimport DevEFO
from clypto.optimizer._native cimport utils as cy
from clypto.optimizer._native import ops
from clypto.optimizer._native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer._native.population cimport NativePopulation
from clypto.optimizer._native.target cimport NativeTarget


cdef class OriginalEFO(DevEFO):
    """
    The original version of: Electromagnetic Field Optimization (EFO)

    Links:
        2. https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + r_rate (float): [0.1, 0.6], default = 0.3, like mutation parameter in GA but for one variable
        + ps_rate (float): [0.5, 0.95], default = 0.85, like crossover parameter in GA
        + p_field (float): [0.05, 0.3], default = 0.1, portion of population, positive field
        + n_field (float): [0.3, 0.7], default = 0.45, portion of population, negative field

    Examples
    ~~~~~~~~
    >>> from clypto.collection.physics_based import EFO    >>> import numpy as np
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
    >>> model = EFO.OriginalEFO(epoch=1000, pop_size=50, r_rate = 0.3, ps_rate = 0.85, p_field = 0.1, n_field = 0.45)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Abedinpourshotorban, H., Shamsuddin, S.M., Beheshti, Z. and Jawawi, D.N., 2016.
    Electromagnetic field optimization: a physics-inspired metaheuristic optimization algorithm.
    Swarm and Evolutionary Computation, 26, pp.8-22.
    """

    cdef public object support_parallel_modes
    cdef public object r_index1
    cdef public object r_index2
    cdef public object r_index3
    cdef public object ps
    cdef public object r_force
    cdef public object rp
    cdef public object randomization
    cdef public object RI

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        r_rate: float = 0.3,
        ps_rate: float = 0.85,
        p_field: float = 0.1,
        n_field: float = 0.45,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (float): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (float): default = 0.85    Like crossover parameter in GA
            p_field (float): default = 0.1     portion of population, positive field
            n_field (float): default = 0.45    portion of population, negative field
        """
        super().__init__(epoch, pop_size, r_rate, ps_rate, p_field, n_field, name=name, mode=mode)
        self.support_parallel_modes = False

    cdef object amend_solution(self, object solution):
        rd = self.generator.uniform(self.problem.lb, self.problem.ub)
        condition = np.logical_and(self.problem.lb <= solution, solution <= self.problem.ub)
        return np.where(condition, solution, rd)

    cdef void initialization(self):
        LegacyNativeOptimizer.initialization(self)
        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        self.r_index1 = self.generator.integers(0, int(self.pop_size * self.p_field), (self.problem.n_dims, self.epoch))
        # random particles from positive field
        self.r_index2 = self.generator.integers(
            int(self.pop_size * (1 - self.n_field)), self.pop_size, (self.problem.n_dims, self.epoch)
        )
        # random particles from negative field
        self.r_index3 = self.generator.integers(
            int((self.pop_size * self.p_field) + 1),
            int(self.pop_size * (1 - self.n_field)),
            (self.problem.n_dims, self.epoch),
        )
        # random particles from neutral field
        self.ps = self.generator.uniform(0, 1, (self.problem.n_dims, self.epoch))
        # Probability of selecting electromagnets of generated particle from the positive field
        self.r_force = self.generator.uniform(0, 1, self.epoch)
        # random force in each generation
        self.rp = self.generator.uniform(0, 1, self.epoch)
        # Some random numbers for checking randomness probability in each generation
        self.randomization = self.generator.uniform(0, 1, self.epoch)
        # Coefficient of randomization when generated electro magnet is out of boundary
        self.RI = 0

    cdef void evolve(self, int epoch):
        cdef NativePopulation pop = self.pop
        cdef NativeTarget tar
        iter01 = epoch - 1
        r = self.r_force[iter01]
        X = pop.X
        cols = np.arange(self.problem.n_dims)
        i1, i2, i3 = self.r_index1[:, iter01], self.r_index2[:, iter01], self.r_index3[:, iter01]
        # electromagnet by electromagnet: one particle of each field per dimension
        x_new = np.where(
            self.ps[:, iter01] > self.ps_rate,
            X[i3, cols] + self.phi * r * (X[i1, cols] - X[i3, cols]) + r * (X[i3, cols] - X[i2, cols]),
            X[i1, cols],
        )
        # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
        if self.rp[iter01] < self.r_rate:
            x_new[self.RI] = (
                    self.problem.lb[self.RI]
                    + (self.problem.ub[self.RI] - self.problem.lb[self.RI])
                    * self.randomization[iter01]
            )
            RI = self.RI + 1
            if RI >= self.problem.n_dims:
                self.RI = 0
        # checking whether the generated number is inside boundary or not
        pos_new = self.correct_solution(x_new)
        tar = self.get_target(pos_new)
        # Updating the population if the fitness of the generated particle is better than worst fitness in
        #     the population (because the population is sorted by fitness, the last particle is the worst)
        ops.set_row(pop, pop.n - 1, pos_new, tar)
