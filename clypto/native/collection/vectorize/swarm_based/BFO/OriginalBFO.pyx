#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 10:21, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport _LegacyAgent


from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalBFO(AgentListOptimizer):
    """
    The original version of: Bacterial Foraging Optimization (BFO)

    Notes:
        + Ned and Nre parameters are replaced by epoch (generation)
        + The Nc parameter will also decrease to reduce the computation time.
        + Cost in this version equal to Fitness value in the paper.
        + https://www.cleveralgorithms.com/nature-inspired/swarm/bfoa.html

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + Ci (float): [0.01, 0.3], step size, default=0.01
        + Ped (float): [0.1, 0.5], probability of elimination, default=0.25
        + Ned (int): elim_disp_steps (Removed), Ned=5,
        + Nre (int): reproduction_steps (Removed), Nre=50,
        + Nc (int): [3, 10], chem_steps (Reduce), Nc = Original Nc/2, default = 5
        + Ns (int): [2, 10], swim length, default=4
        + d_attract (float): coefficient to calculate attract force, default = 0.1
        + w_attract (float): coefficient to calculate attract force, default = 0.2
        + h_repels (float): coefficient to calculate repel force, default = 0.1
        + w_repels (float): coefficient to calculate repel force, default = 10

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.swarm_based import BFO    >>> import numpy as np
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
    >>> model = BFO.OriginalBFO(epoch=1000, pop_size=50, Ci = 0.01, Ped = 0.25, Nc = 5, Ns = 4, d_attract=0.1, w_attract=0.2, h_repels=0.1, w_repels=10)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Passino, K.M., 2002. Biomimicry of bacterial foraging for distributed optimization and control.
    IEEE control systems magazine, 22(3), pp.52-67.
    """

    cdef public object d_attract
    cdef public object w_attract
    cdef public object h_repels
    cdef public object w_repels
    cdef public object step_size
    cdef public object Ci
    cdef public object p_eliminate
    cdef public object Ped
    cdef public object chem_steps
    cdef public object Nc
    cdef public object swim_length
    cdef public object Ns
    cdef public object half_pop_size

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        Ci: float = 0.01,
        Ped: float = 0.25,
        Nc: int = 5,
        Ns: int = 4,
        d_attract: float = 0.1,
        w_attract: float = 0.2,
        h_repels: float = 0.1,
        w_repels: float = 10,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            Ci (float): step size, default=0.01
            Ped (float): p_eliminate, default=0.25
            Ned (int): elim_disp_steps (Removed)         Ned=5,
            Nre (int): reproduction_steps (Removed)      Nre=50,
            Nc (int): chem_steps (Reduce)                Nc = Original Nc/2, default = 5
            Ns (int): swim_length, default=4
            d_attract (float): coefficient to calculate attract force, default = 0.1
            w_attract (float): coefficient to calculate attract force, default = 0.2
            h_repels (float): coefficient to calculate repel force, default = 0.1
            w_repels (float): coefficient to calculate repel force, default = 10
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=[
                "epoch",
                "pop_size",
                "Ci",
                "Ped",
                "Nc",
                "Ns",
                "d_attract",
                "w_attract",
                "h_repels",
                "w_repels",
            ],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.step_size = self.Ci = cy.validator(int, Ci, (0, 5.0), "Ci")
        self.p_eliminate = self.Ped = cy.validator(float, Ped, (0, 1.0), "Ped")
        self.chem_steps = self.Nc = cy.validator(int, Nc, [2, 100], "Nc")
        self.swim_length = self.Ns = cy.validator(int, Ns, [2, 100], "Ns")
        self.d_attract = cy.validator(float, d_attract, (0, 1.0), "d_attract")
        self.w_attract = cy.validator(float, w_attract, (0, 1.0), "w_attract")
        self.h_repels = cy.validator(float, h_repels, (0, 1.0), "h_repels")
        self.w_repels = cy.validator(float, w_repels, (2.0, 20.0), "w_repels")
        self.half_pop_size = int(self.pop_size / 2)

    def generate_empty_agent(self, solution: np.ndarray | None = None) -> _LegacyAgent:
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        cost = 0.0
        interaction = 0.0
        nutrients = 0.0
        return FieldAgent(
            solution=solution, cost=cost, interaction=interaction, nutrients=nutrients
        )

    def compute_cell_interaction__(self, cell, cells, d, w):
        sum_inter = 0.0
        for other in cells:
            diff = self.problem.n_dims * ((cell.solution - other.solution) ** 2).mean(
                axis=None
            )
            sum_inter += d * np.exp(w * diff)
        return sum_inter

    def attract_repel__(self, idx, cells):
        attract = self.compute_cell_interaction__(
            cells[idx], cells, -self.d_attract, -self.w_attract
        )
        repel = self.compute_cell_interaction__(
            cells[idx], cells, self.h_repels, -self.w_repels
        )
        return attract + repel

    def evaluate__(self, idx, cells):
        cells[idx].interaction = self.attract_repel__(idx, cells)
        cells[idx].cost = cells[idx].target.fitness + cells[idx].interaction
        return cells

    def tumble_cell__(self, cell, step_size):
        delta_i = self.generator.uniform(self.problem.lb, self.problem.ub)
        unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
        vector = cell.solution + step_size * unit_vector
        return [vector, 0.0, 0.0, 0.0, 0.0]

    def evolve_agents(self, epoch):
        for j in range(0, self.chem_steps):
            for idx in range(0, self.pop_size):
                sum_nutrients = 0.0
                self.objs = self.evaluate__(idx, self.objs)
                sum_nutrients += self.objs[idx].cost

                for m in range(0, self.swim_length):
                    delta_i = self.generator.uniform(self.problem.lb, self.problem.ub)
                    unit_vector = delta_i / np.sqrt(np.abs(np.dot(delta_i, delta_i.T)))
                    pos_new = self.objs[idx].solution + self.step_size * unit_vector
                    pos_new = self.correct_solution(pos_new)
                    agent = self.generate_agent(pos_new)
                    if self.compare_target(
                        agent.target, self.objs[idx].target, self.problem.minmax
                    ):
                        self.objs[idx] = agent
                        break
                    sum_nutrients += self.objs[idx].cost
                self.objs[idx].nutrients = sum_nutrients
            cells = sorted(self.objs, key=lambda cell: cell.nutrients)
            self.objs = (
                cells[0 : self.half_pop_size].copy()
                + cells[0 : self.pop_size - self.half_pop_size].copy()
            )
            for idc in range(self.pop_size):
                if self.generator.random() < self.p_eliminate:
                    self.objs[idc] = self.generate_agent()
