#!/usr/bin/env python
# cython: boundscheck=True
# (classic list code: out-of-range indexing raises IndexError instead of crashing)
# Created by "Thieu" at 00:47, 16/10/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.optimizer cimport LegacyNativeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent_list cimport AgentListOptimizer
from clypto.optimizer.native.agent_list import FieldAgent


cdef class OriginalHBO(AgentListOptimizer):
    """
    The original version of: Heap-based optimizer (HBO)

    Links:
        1. https://www.sciencedirect.com/science/article/abs/pii/S0957417420305261#!
        2. https://github.com/qamar-askari/HBO/blob/master/HBO.m

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + degree (int): [2, 4], the degree level in Corporate Rank Hierarchy (CRH), default=2

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.human_based import HBO    >>> import numpy as np
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
    >>> model = HBO.OriginalHBO(epoch=1000, pop_size=50, degree = 3)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Askari, Q., Saeed, M., & Younas, I. (2020). Heap-based optimizer inspired by corporate rank hierarchy
    for global optimization. Expert Systems with Applications, 161, 113702.
    """

    cdef public object degree
    cdef public object cycles
    cdef public object it_per_cycle
    cdef public object qtr_cycle
    cdef public object heap
    cdef public object friend_limits

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        degree: int = 2,
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            degree (int): the degree level in Corporate Rank Hierarchy (CRH), default=2
        """
        LegacyNativeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "degree"],
            sort_flag=False,
            parallelizable=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.degree = cy.validator(int, degree, [2, 10], "degree")

    cdef void initialize_variables(self):
        self.cycles = np.floor(self.epoch / 25)
        self.it_per_cycle = self.epoch / self.cycles
        self.qtr_cycle = self.it_per_cycle / 4

    def colleagues_limits_generator__(self, pop_size, degree=3):
        friend_limits = np.zeros((pop_size, 2))
        for c in range(pop_size - 1, -1, -1):
            hi = int(np.ceil((np.log10(c * degree - c + 1) / np.log10(degree)))) - 1
            lowerLim = (degree * degree ** (hi - 1) - 1) / (degree - 1) + 1
            upperLim = (degree * degree ** hi - 1) / (degree - 1)
            friend_limits[c, 0] = lowerLim if lowerLim <= pop_size else pop_size
            friend_limits[c, 1] = upperLim if upperLim <= pop_size else pop_size
        return friend_limits.astype(int)

    def heapifying__(self, pop, degree=3):
        pop_size = len(pop)
        heap = []
        for c in range(pop_size):
            heap.append([pop[c].target, c])
            # Heapifying
            t = c
            while t > 0:
                parent_id = int(np.floor((t + 1) / degree) - 1)
                if self.compare_target(
                        pop[parent_id].target, pop[t].target, self.problem.minmax
                ):
                    break
                else:
                    heap[t], heap[parent_id] = heap[parent_id], heap[t]
                t = parent_id
        return heap

    cdef void before_main_loop(self):
        self.heap = self.heapifying__(self.objs, self.degree)
        self.friend_limits = self.colleagues_limits_generator__(
            self.pop_size, self.degree
        )

    def evolve_agents(self, epoch):
        gama = (np.mod(epoch, self.it_per_cycle) + 1) / self.qtr_cycle
        gama = np.abs(2 - gama)
        p1 = 1.0 - epoch / self.epoch
        p2 = p1 + (1 - p1) / 2
        for c in range(self.pop_size - 1, 0, -1):
            if c == 0:  # Dealing with root
                continue
            else:
                parent_id = int(np.floor((c + 1) / self.degree) - 1)
                cur_agent = self.objs[self.heap[c][1]].copy()  # Sol to be updated
                par_agent = self.objs[
                    self.heap[parent_id][1]
                ]  # Sol to be updated with reference to
                # Sol to be updated with reference to
                if self.friend_limits[c, 0] < self.friend_limits[c, 1] + 1:
                    friend_idx = self.friend_limits[c, 0]
                else:
                    friend_idx = self.generator.choice(
                        list(
                            set(
                                range(
                                    self.friend_limits[c, 0], self.friend_limits[c, 1]
                                )
                            )
                            - {c}
                        )
                    )
                fri_agent = self.objs[self.heap[friend_idx][1]]
                # Position Updating
                rr = self.generator.random(self.problem.n_dims)
                rn = 2 * self.generator.random(self.problem.n_dims) - 1
                for jdx in range(self.problem.n_dims):
                    if rr[jdx] < p1:
                        continue
                    elif rr[jdx] < p2:
                        cur_agent.solution[jdx] = par_agent.solution[jdx] + rn[
                            jdx
                        ] * gama * np.abs(
                            par_agent.solution[jdx] - cur_agent.solution[jdx]
                        )
                    else:
                        if self.compare_target(
                                self.heap[friend_idx][0],
                                self.heap[c][0],
                                self.problem.minmax,
                        ):
                            cur_agent.solution[jdx] = fri_agent.solution[jdx] + rn[
                                jdx
                            ] * gama * np.abs(
                                fri_agent.solution[jdx] - cur_agent.solution[jdx]
                            )
                        else:
                            cur_agent.solution[jdx] += (
                                    rn[jdx]
                                    * gama
                                    * np.abs(
                                fri_agent.solution[jdx] - cur_agent.solution[jdx]
                            )
                            )
                pos_new = self.correct_solution(cur_agent.solution)
                cur_agent = self.generate_agent(pos_new)
                if self.compare_target(
                        cur_agent.target, self.heap[c][0], self.problem.minmax
                ):
                    self.objs[self.heap[c][1]] = cur_agent
                    self.heap[c][0] = cur_agent.target.copy()
            # Heapifying
            t = c
            while t > 1:
                parent_id = int((t + 1) / self.degree)
                if self.compare_target(
                        self.heap[parent_id][0], self.heap[t][0], self.problem.minmax
                ):
                    break
                else:
                    self.heap[t], self.heap[parent_id] = (
                        self.heap[parent_id],
                        self.heap[t],
                    )
                t = parent_id
