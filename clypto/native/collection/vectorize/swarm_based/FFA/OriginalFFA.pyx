#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 17:13, 01/03/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalFFA(AgentListOptimizer):
    """
    The original version of: Firefly Algorithm (FFA)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + gamma (float): Light Absorption Coefficient, default = 0.001
        + beta_base (float): Attraction Coefficient Base Value, default = 2
        + alpha (float): Mutation Coefficient, default = 0.2
        + alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
        + delta (float): Mutation Step Size, default = 0.05
        + exponent (int): Exponent (m in the paper), default = 2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import FFA    >>> import numpy as np
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
    >>> model = FFA.OriginalFFA(epoch=1000, pop_size=50, gamma = 0.001, beta_base = 2, alpha = 0.2, alpha_damp = 0.99, delta = 0.05, exponent = 2)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Gandomi, A.H., Yang, X.S. and Alavi, A.H., 2011. Mixed variable structural optimization
    using firefly algorithm. Computers & Structures, 89(23-24), pp.2325-2336.
    [2] Arora, S. and Singh, S., 2013. The firefly optimization algorithm: convergence analysis and
    parameter selection. International Journal of Computer Applications, 69(3).
    """

    cdef public object gamma
    cdef public object beta_base
    cdef public object alpha
    cdef public object alpha_damp
    cdef public object delta
    cdef public object exponent
    cdef public object dyn_alpha

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        gamma: float = 0.001,
        beta_base: float = 2,
        alpha: float = 0.2,
        alpha_damp: float = 0.99,
        delta: float = 0.05,
        exponent: int = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            gamma (float): Light Absorption Coefficient, default = 0.001
            beta_base (float): Attraction Coefficient Base Value, default = 2
            alpha (float): Mutation Coefficient, default = 0.2
            alpha_damp (float): Mutation Coefficient Damp Rate, default = 0.99
            delta (float): Mutation Step Size, default = 0.05
            exponent (int): Exponent (m in the paper), default = 2
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "gamma",
                "beta_base",
                "alpha",
                "alpha_damp",
                "delta",
                "exponent",
            ],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.gamma = cy.validator(float, gamma, (0, 1.0), "gamma")
        self.beta_base = cy.validator(float, beta_base, (0, 3.0), "beta_base")
        self.alpha = cy.validator(float, alpha, (0, 1.0), "alpha")
        self.alpha_damp = cy.validator(float, alpha_damp, (0, 1.0), "alpha_damp")
        self.delta = cy.validator(float, delta, (0, 1.0), "delta")
        self.exponent = cy.validator(int, exponent, [2, 4], "exponent")

    def _initialize_variables(self):
        self.dyn_alpha = self.alpha  # Initial Value of Mutation Coefficient

    def _evolve_agents(self, epoch):
        # Maximum Distance
        dmax = np.sqrt(self.problem.n_dims)
        for idx in range(0, self.pop_size):
            agent = self.objs[idx].copy()
            pop_child = []
            for j in range(idx + 1, self.pop_size):
                # Move Towards Better Solutions
                if self._compare_target(
                        self.objs[j].target, agent.target, self.problem.sense
                ):
                    # Calculate Radius and Attraction Level
                    rij = np.linalg.norm(agent.solution - self.objs[j].solution) / dmax
                    beta = self.beta_base * np.exp(-self.gamma * rij ** self.exponent)
                    # Mutation Vector
                    mutation_vector = self.delta * self.generator.uniform(
                        0, 1, self.problem.n_dims
                    )
                    temp = np.matmul(
                        (self.objs[j].solution - agent.solution),
                        self.generator.uniform(
                            0, 1, (self.problem.n_dims, self.problem.n_dims)
                        ),
                    )
                    pos_new = (
                            agent.solution + self.dyn_alpha * mutation_vector + beta * temp
                    )
                    pos_new = self._correct_solution(pos_new)
                    agent = self._generate_agent(pos_new)
                    pop_child.append(agent)
            if len(pop_child) < self.pop_size:
                pop_child += self._generate_agents(self.pop_size - len(pop_child))
            local_best = self._get_best_agent(pop_child, self.problem.sense)
            # Compare to Previous Solution
            if self._compare_target(
                    local_best.target, agent.target, self.problem.sense
            ):
                self.objs[idx] = local_best
        self.objs.append(self.g_best)
        self.dyn_alpha = self.alpha_damp * self.alpha
