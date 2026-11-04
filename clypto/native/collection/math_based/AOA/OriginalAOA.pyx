cimport clypto.core as cy
#!/usr/bin/env python
# Created by "Thieu" at 09:56, 07/07/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%



cdef class OriginalAOA(cy.Optimizer):
    """
    The original version of: Arithmetic Optimization Algorithm (AOA)

    Links:
        1. https://doi.org/10.1016/j.cma.2020.113609

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + alpha (int): [3, 8], fixed parameter, sensitive exploitation parameter, Default: 5,
        + miu (float): [0.3, 1.0], fixed parameter , control parameter to adjust the search process, Default: 0.5,
        + moa_min (float): [0.1, 0.4], range min of Math Optimizer Accelerated, Default: 0.2,
        + moa_max (float): [0.5, 1.0], range max of Math Optimizer Accelerated, Default: 0.9,

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.math_based import AOA    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "sense": "min",
    >>>     "obj_func": objective_function
    >>> }
    >>>
    >>> model = AOA.OriginalAOA(epoch=1000, pop_size=50, alpha = 5, miu = 0.5, moa_min = 0.2, moa_max = 0.9)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] Abualigah, L., Diabat, A., Mirjalili, S., Abd Elaziz, M. and Gandomi, A.H., 2021. The arithmetic
    optimization algorithm. Computer methods in applied mechanics and engineering, 376, p.113609.
    """

    cdef public int alpha
    cdef public double miu
    cdef public double moa_max
    cdef public double moa_min

    def __init__(
            self,
            epoch: int = 10000,
            pop_size: int = 100,
            alpha: float = 5,
            miu: float = 0.5,
            moa_min: float = 0.2,
            moa_max: float = 0.9,
            **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): fixed parameter, sensitive exploitation parameter, Default: 5,
            miu (float): fixed parameter, control parameter to adjust the search process, Default: 0.5,
            moa_min (float): range min of Math Optimizer Accelerated, Default: 0.2,
            moa_max (float): range max of Math Optimizer Accelerated, Default: 0.9,
        """
        super().__init__(parameters=["epoch", "pop_size", "alpha", "miu", "moa_min", "moa_max"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])
        self.alpha = cy.validator(int, alpha, [2, 10], "alpha")
        self.miu = cy.validator(float, miu, [0.1, 2.0], "miu")
        self.moa_min = cy.validator(float, moa_min, (0, 0.41), "moa_min")
        self.moa_max = cy.validator(float, moa_max, (0.41, 1.0), "moa_max")

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        moa = self.moa_min + epoch * (
                (self.moa_max - self.moa_min) / self.epoch
        )  # Eq. 2
        mop = 1 - ((epoch + 1) ** (1.0 / self.alpha)) / (
                self.epoch ** (1.0 / self.alpha)
        )  # Eq. 4
        cdef cy.Population n_population = cy.empty_snapshot(self.population)
        for idx, agent in enumerate(self.population.toarray()):
            x = agent.solution.copy()
            for j in range(0, self.problem.n_dims):
                r1, r2, r3 = self.generator.random(3)
                if r1 > moa:  # Exploration phase
                    if r2 < 0.5:
                        x[j] = (
                                self.g_best.solution[j]
                                / (mop + self.EPSILON)
                                * (
                                        (self.problem.bounds.up[j] - self.problem.bounds.low[j]) * self.miu
                                        + self.problem.bounds.low[j]
                                )
                        )
                    else:
                        x[j] = (
                                self.g_best.solution[j]
                                * mop
                                * (
                                        (self.problem.bounds.up[j] - self.problem.bounds.low[j]) * self.miu
                                        + self.problem.bounds.low[j]
                                )
                        )
                else:  # Exploitation phase
                    if r3 < 0.5:
                        x[j] = self.g_best.solution[j] - mop * (
                                (self.problem.bounds.up[j] - self.problem.bounds.low[j]) * self.miu
                                + self.problem.bounds.low[j]
                        )
                    else:
                        x[j] = self.g_best.solution[j] + mop * (
                                (self.problem.bounds.up[j] - self.problem.bounds.low[j]) * self.miu
                                + self.problem.bounds.low[j]
                        )
            x = cy.correct_solution(self.problem, x)
            child = self.population.create_agent(x)
            n_population.append(child)
        self.population = self.population.greedy(self.population.evaluate(n_population, self.mode), self.mode)
