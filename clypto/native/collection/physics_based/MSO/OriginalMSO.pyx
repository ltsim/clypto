#!/usr/bin/env python
# Created by "Thieu" at 16:31, 13/09/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
cimport clypto.core as cy



cdef class OriginalMSO(cy.Optimizer):
    """
    The original version of: Mirage Search Optimization (MSO)

    Links:
        1. https://www.mathworks.com/matlabcentral/fileexchange/180042-mirage-search-optimization

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.physics_based import MSO    >>> import numpy as np
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
    >>> model = MSO.OriginalMSO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.fitness}")

    References
    ~~~~~~~~~~
    [1] He, J., Zhao, S., Ding, J., & Wang, Y. (2025). Mirage search optimization: Application to
    path planning and engineering design problems. Advances in Engineering Software, 203, 103883.
    """

    def __init__(
            self, epoch: int = 10000, pop_size: int = 100, **kwargs: object
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
        """
        super().__init__(parameters=["epoch", "pop_size"], sort_flag=False, **kwargs)
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.population = cy.population(pop_size, range=[10, 10000])

    def sind(self, x):
        return np.sin(np.deg2rad(x))

    def cosd(self, x):
        return np.cos(np.deg2rad(x))

    def tand(self, x):
        x = np.asarray(x, dtype=float)
        bad = (np.mod(x, 90) == 0) & (np.mod(x, 180) != 0)
        x[bad] = x[bad] - self.EPSILON
        return np.tan(np.deg2rad(x))

    def atand(self, x):
        return np.rad2deg(np.arctan(x))

    def asind(self, x):
        x = np.clip(x, -1, 1)  # asin bound [-1, 1]
        val = np.rad2deg(np.arcsin(x))
        return val

    def atanh(self, x):
        if np.abs(x) >= 1:
            return 1.0
        return np.arctanh(x)

    cdef void evolve(self, int epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_size = self.population.size()
        # Random permutation for agent selection
        ac = self.generator.permutation(pop_size - 1) + 1
        # Selection of individuals for Superior mirage search
        cv = int(
            np.ceil(
                (pop_size * (2 / 3))
                * ((self.epoch - self.nf_counter + 1) / self.epoch)
            )
        )

        # Superior mirage search
        pop_new = []
        for idx in ac[:cv]:
            pos_new = np.zeros(self.problem.n_dims)
            for k in range(self.problem.n_dims):
                h = (
                            self.g_best.solution[k] - self.population[idx].solution[k]
                    ) * self.generator.random()
                cmax = 1
                hmax = 5 * self.atanh(-(self.nf_counter / self.epoch) + 1) + cmax
                if h > hmax:
                    h = hmax
                if h < cmax:
                    h = cmax
                zf = self.generator.choice([-1, 1])
                a = self.generator.random() * 20
                b = self.generator.random() * (45 - a / 2)
                z = self.generator.integers(1, 3)
                A = B = C = D = 90
                if z == 1:
                    C = b + 90
                    D = 180 - C - a
                    B = 180 - 2 * D
                    A = 180 - B + a - 90
                elif z == 2 and a < b:
                    C = 90 - b
                    D = 90 + a - b
                    B = 180 - 2 * D
                    A = 180 - B - a - 90
                elif z == 2 and a > b:
                    C = 90 - b
                    D = 180 - C - a
                    B = 180 - 2 * D
                    A = 180 - B - 90 + a
                else:
                    zf = 0
                dx = (self.sind(B) * h * self.sind(C)) / (self.sind(D) * self.sind(A))
                dx = dx * zf
                pos_new[k] = self.population[idx].solution[k] + dx
            # Bound the variables
            pos_new = cy.correct_solution(self.problem, pos_new)
            agent = self.population.generate_agent(pos_new)
            pop_new.append(agent)
        self.population = self.population.spawn(cy.sort_agents(self.population + pop_new, self.problem.sense)[:pop_size])

        # Inferior mirage search
        pop_new = []
        for idx, agent in enumerate(self.population.toarray()):
            if self.g_best == agent:
                hh = (
                        np.ones(self.problem.n_dims) * 0.05 * self.generator.choice([-1, 1])
                )
            else:
                hh = self.g_best.solution - agent.solution
            zf = np.sign(hh)
            hh = np.abs(hh * self.generator.random(self.problem.n_dims))
            gama = (
                    self.generator.random(self.problem.n_dims)
                    * 90
                    * ((self.epoch - self.nf_counter * 0.99) / self.epoch)
            )
            amax = self.atand(1.0 / (2 * self.tand(gama)))
            amin = self.atand(
                (self.sind(gama) * self.cosd(gama)) / (1 + (self.sind(gama)) ** 2)
            )
            fai = (amax - amin) * self.generator.random() + amin
            omg = self.asind(self.generator.random() * self.sind(fai + gama))
            x = (hh / self.tand(gama)) - (
                    (
                            (hh / self.sind(gama))
                            - (hh * self.sind(fai)) / (self.cosd(fai + gama))
                    )
                    * self.cosd(omg)
            ) / self.cosd(omg - gama)
            pos_new = agent.solution + x * zf
            pos_new = cy.correct_solution(self.problem, pos_new)
            child = self.population.generate_agent(pos_new)
            pop_new.append(child)
        self.population = self.population.spawn(cy.sort_agents(self.population + pop_new, self.problem.sense)[:pop_size])
