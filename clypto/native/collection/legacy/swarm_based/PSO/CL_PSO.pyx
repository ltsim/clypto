#!/usr/bin/env python
# Created by "Thieu" at 09:49, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np

from clypto.optimizer.native.agent cimport LegacyAgent
from clypto.optimizer.native.legacy cimport LegacyOptimizer


cdef class _CL_PSOAgent(LegacyAgent):
    cdef public object velocity
    cdef public object local_solution
    cdef public object local_target


cdef class CL_PSO(LegacyOptimizer):
    """
    The original version of: Comprehensive Learning Particle Swarm Optimization (CL-PSO)

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + c_local (float): [1.0, 3.0], local coefficient, default = 1.2
        + w_min (float): [0.1, 0.5], Weight min of bird, default = 0.4
        + w_max (float): [0.7, 2.0], Weight max of bird, default = 0.9
        + max_flag (int): [5, 20], Number of times, default = 7

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.legacy.swarm_based import PSO    >>> import numpy as np
    >>> from clypto import NumberBounds
    >>>
    >>> def objective_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict = {
    >>>     "bounds": NumberBounds(float, low=(-10.,) * 30, up=(10.,) * 30, name="delta"),
    >>>     "obj_func": objective_function,
    >>>     "sense": "min",
    >>> }
    >>>
    >>> model = PSO.CL_PSO(epoch=1000, pop_size=50, c_local = 1.2, w_min=0.4, w_max=0.9, max_flag = 7)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Liang, J.J., Qin, A.K., Suganthan, P.N. and Baskar, S., 2006. Comprehensive learning particle swarm optimizer
    for global optimization of multimodal functions. IEEE transactions on evolutionary computation, 10(3), pp.281-295.
    """

    def __init__(
        self,
        epoch: int = 10000,
        pop_size: int = 100,
        c_local: float = 1.2,
        w_min: float = 0.4,
        w_max: float = 0.9,
        max_flag: int = 7,
        **kwargs: object
    ) -> None:
        """
        Args:
            epoch: maximum number of iterations, default = 10000
            pop_size: number of population size, default = 100
            c_local: local coefficient, default = 1.2
            w_min: Weight min of bird, default = 0.4
            w_max: Weight max of bird, default = 0.9
            max_flag: Number of times, default = 7
        """
        LegacyOptimizer.__init__(self, **kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.c_local = self.validator.check_float("c_local", c_local, (0, 5.0))
        self.w_min = self.validator.check_float("w_min", w_min, (0, 0.5))
        self.w_max = self.validator.check_float("w_max", w_max, [0.5, 2.0])
        self.max_flag = self.validator.check_int("max_flag", max_flag, [2, 100])
        self._set_parameters(
            ["epoch", "pop_size", "c_local", "w_min", "w_max", "max_flag"]
        )
        self.sort_flag = False

    def _initialize_variables(self):
        self.v_max = 0.5 * (self.problem.bounds.up - self.problem.bounds.low)
        self.v_min = -self.v_max
        self.flags = np.zeros(self.pop_size)

    def _generate_empty_agent(self, solution: np.ndarray | None = None):
        if solution is None:
            solution = self.problem.generate_solution(encoded=True)
        velocity = self.generator.uniform(-self.v_max, self.v_max)
        local_pos = solution.copy()
        return _CL_PSOAgent(
            solution=solution, velocity=velocity, local_solution=local_pos
        )

    def _generate_agent(self, solution: np.ndarray | None = None):
        agent = self._generate_empty_agent(solution)
        agent.target = self._get_target(agent.solution)
        agent.local_target = agent.target.copy()
        return agent

    def _evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from LegacyOptimizer class

        Args:
            epoch (int): The current iteration
        """
        wk = self.w_max * (epoch / self.epoch) * (self.w_max - self.w_min)
        pop_new = []
        for idx in range(0, self.pop_size):
            pci = 0.05 + 0.45 * (np.exp(10 * (idx + 1) / self.pop_size) - 1) / (
                np.exp(10) - 1
            )
            vec_new = self.pop[idx].velocity.copy()
            for jdx in range(0, self.problem.n_dims):
                if self.generator.random() > pci:
                    vj = wk * self.pop[idx].velocity[
                        jdx
                    ] + self.c_local * self.generator.random() * (
                        self.pop[idx].local_solution[jdx] - self.pop[idx].solution[jdx]
                    )
                else:
                    id1, id2 = self.generator.choice(
                        list(set(range(0, self.pop_size)) - {idx}), 2, replace=False
                    )
                    if self._compare_target(
                        self.pop[id1].target, self.pop[id2].target, self.problem.sense
                    ):
                        vj = wk * self.pop[idx].velocity[
                            jdx
                        ] + self.c_local * self.generator.random() * (
                            self.pop[id1].local_solution[jdx]
                            - self.pop[idx].solution[jdx]
                        )
                    else:
                        vj = wk * self.pop[idx].velocity[
                            jdx
                        ] + self.c_local * self.generator.random() * (
                            self.pop[id2].local_solution[jdx]
                            - self.pop[idx].solution[jdx]
                        )
                vec_new[jdx] = vj
            vec_new = np.clip(vec_new, self.v_min, self.v_max)
            pos_new = self.pop[idx].solution + vec_new
            pos_new = self._correct_solution(pos_new)

            pos_new = self._correct_solution(pos_new)
            agent = self._generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self._get_target(pos_new)
                agent.local_target = agent.target.copy()
                self.pop[idx] = self._get_better_agent(
                    self.pop[idx], agent, self.problem.sense
                )
                if self._compare_target(
                    agent.target, self.pop[idx].local_target, self.problem.sense
                ):
                    self.pop[idx].update(
                        local_solution=agent.solution.copy(),
                        local_target=agent.target.copy(),
                    )
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1
                    if self.flags[idx] >= self.max_flag:
                        self.flags[idx] = 0
        if self.mode in self.AVAILABLE_MODES:
            pop_new = self._update_target_for_population(pop_new)
            pop_child = self._greedy_selection_population(
                self.pop, pop_new, self.problem.sense
            )
            for idx in range(0, self.pop_size):
                if self._compare_target(
                    pop_new[idx].target, self.pop[idx].local_target, self.problem.sense
                ):
                    pop_child[idx].update(
                        local_solution=pop_new[idx].solution.copy(),
                        local_target=pop_new[idx].target.copy(),
                    )
                    self.flags[idx] = 0
                else:
                    self.flags[idx] += 1
                    if self.flags[idx] >= self.max_flag:
                        self.flags[idx] = 0
            self.pop = pop_child
