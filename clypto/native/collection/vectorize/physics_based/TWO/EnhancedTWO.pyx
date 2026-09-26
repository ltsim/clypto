#!/usr/bin/env python
# Created by "Thieu" at 21:18, 17/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np

from clypto.native.collection.vectorize.physics_based.TWO.OriginalTWO cimport OriginalTWO
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation


class _TWOAgent:
    """A classic team (solution, target, weight).

    Enhanced TWO shares agents between its lists and edits them in place, and its initial
    population holds every agent twice; those aliasing effects are part of the classic
    behaviour, so this algorithm keeps agent objects and mirrors them into ``self.pop``.
    """

    __slots__ = ("solution", "target", "weight")

    def __init__(self, solution, target, weight=0.0):
        self.solution, self.target, self.weight = solution, target, weight

    def copy(self):
        return _TWOAgent(self.solution, self.target.copy(), self.weight)


cdef class EnhancedTWO(OriginalTWO):
    """
    The original version of: Enhenced Tug of War Optimization (ETWO)

    Links:
        1. https://doi.org/10.1016/j.procs.2020.03.063

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.physics_based import TWO    >>> import numpy as np
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
    >>> model = TWO.EnhancedTWO(epoch=1000, pop_size=50)
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Nguyen, T., Hoang, B., Nguyen, G. and Nguyen, B.M., 2020. A new workload prediction model using
    extreme learning machine and enhanced tug of war optimization. Procedia Computer Science, 170, pp.362-369.
    """

    cdef public object objs

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
        super().__init__(epoch, pop_size, name=name, mode=mode)

    def update_target__(self, agents):
        # update_target_for_population: swarm modes evaluate every agent (counted once per agent)
        for agent in agents:
            agent.target = self._get_target(agent.solution, counted=False)
        self._nfe_counter += len(agents)

    def better_agent__(self, x, y):
        # get_better_agent(x, y)
        if self.problem.sense == "min":
            return x.copy() if x.target.fitness < y.target.fitness else y.copy()
        return y.copy() if x.target.fitness < y.target.fitness else x.copy()

    def update_weight_agents__(self, teams):
        list_fits = np.array([agent.target.fitness for agent in teams])
        maxx, minn = np.max(list_fits), np.min(list_fits)
        if maxx == minn:
            list_fits = self.generator.uniform(0.0, 1.0, self.pop_size)
        list_weights = np.exp(-(list_fits - maxx) / (maxx - minn))
        list_weights = list_weights / np.sum(list_weights) + 0.1
        for idx in range(self.pop_size):
            teams[idx].weight = list_weights[idx]
        return teams

    def mirror__(self):
        """Copy the agents into the population the engine sorts and reports."""
        cdef NativePopulation pop = self.pop.take(np.zeros(len(self.objs), dtype=int))
        for i, agent in enumerate(self.objs):
            pop.X[i] = agent.solution
            pop.O[i] = agent.target.objectives
            pop.F[i] = agent.target.fitness
            pop.field("W")[i, 0] = agent.weight
        return pop

    def _initialization(self):
        cdef NativePopulation base
        VectorizeOptimizer._initialization(self)
        base = self.pop
        pop = [_TWOAgent(base.X[i].copy(), base.agent(i).target) for i in range(base.n)]
        pop_oppo = list(pop)  # pop.copy(): the very same agents
        for idx in range(self.pop_size):
            pos_opposite = self.problem.bounds.up + self.problem.bounds.low - pop[idx].solution
            pos_new = self._correct_solution(pos_opposite)
            pop_oppo[idx].solution = pos_new
            if self.mode not in self.AVAILABLE_MODES:
                pop_oppo[idx].target = self._get_target(pos_new)
        if self.mode in self.AVAILABLE_MODES:
            self.update_target__(pop_oppo)
        merged = pop + pop_oppo
        order = np.argsort([agent.target.fitness for agent in merged])
        if self.problem.sense == "max":
            order = order[::-1]
        self.objs = self.update_weight_agents__([merged[i] for i in order][:self.pop_size])
        self.pop = self.mirror__()

    def _evolve(self, int epoch_c):
        cdef object epoch = epoch_c
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        sense = self.problem.sense
        lb, ub = self.problem.bounds.low, self.problem.bounds.up
        pop = list(self.objs)
        pop_new = list(pop)  # self.pop.copy()
        # g_best is one of the agents (aliased) once the first epoch is over, a copy before
        g_best = self.objs[self._g_best_row] if self._g_best_row >= 0 else self.g_best
        for idx in range(self.pop_size):
            pos_new = pop[idx].solution.copy().astype(float)
            for kdx in range(self.pop_size):
                if pop[idx].weight < pop[kdx].weight:
                    force = max(pop[idx].weight * self.muy_s, pop[kdx].weight * self.muy_s)
                    resultant_force = force - pop[idx].weight * self.muy_k
                    g = pop[kdx].solution - pop[idx].solution
                    acceleration = resultant_force * g / (pop[idx].weight * self.muy_k)
                    delta_x = 0.5 * acceleration + np.power(self.alpha, epoch) * self.beta * (
                        ub - lb
                    ) * self.generator.normal(0, 1, self.problem.n_dims)
                    pos_new += delta_x
            pop_new[idx].solution = pos_new
        for idx in range(self.pop_size):
            pos_new = pop[idx].solution.copy().astype(float)
            for jdx in range(self.problem.n_dims):
                if pos_new[jdx] < lb[jdx] or pos_new[jdx] > ub[jdx]:
                    if self.generator.random() <= 0.5:
                        pos_new[jdx] = g_best.solution[jdx] + self.generator.standard_normal() / epoch * (
                            g_best.solution[jdx] - pos_new[jdx]
                        )
                        if pos_new[jdx] < lb[jdx] or pos_new[jdx] > ub[jdx]:
                            pos_new[jdx] = pop[idx].solution[jdx]
                    else:
                        if pos_new[jdx] < lb[jdx]:
                            pos_new[jdx] = lb[jdx]
                        if pos_new[jdx] > ub[jdx]:
                            pos_new[jdx] = ub[jdx]
            pop_new[idx].solution = self._correct_solution(pos_new)
            if not swarm:
                pop_new[idx].target = self._get_target(pos_new)
                pop[idx] = self.better_agent__(pop_new[idx], pop[idx])
        if swarm:
            self.update_target__(pop_new)
            # greedy_selection_population(pop, pop_new)
            if sense == "min":
                pop = [pop_new[i] if pop_new[i].target.fitness < pop[i].target.fitness else pop[i] for i in range(len(pop))]
            else:
                pop = [pop_new[i] if pop_new[i].target.fitness > pop[i].target.fitness else pop[i] for i in range(len(pop))]

        for idx in range(self.pop_size):
            # generate_opposition_solution(pop_new[idx], g_best)
            C_op = self._correct_solution(
                lb + ub - g_best.solution + self.generator.uniform() * (g_best.solution - pop_new[idx].solution)
            )
            pos_new = self._correct_solution(C_op)
            agent = _TWOAgent(pos_new, self._get_target(pos_new))
            if self._compare_fitness(agent.target.fitness, pop_new[idx].target.fitness, sense):
                pop_new[idx] = agent
            else:
                levy_step = self._get_levy_flight_step(beta=1.0, multiplier=1.0, size=self.problem.n_dims, case=-1)
                pos_new = pop_new[idx].solution + 1.0 / np.sqrt(epoch) * levy_step
                pos_new = self._correct_solution(pos_new)
                agent = _TWOAgent(pos_new, self._get_target(pos_new))
                if self._compare_fitness(agent.target.fitness, pop_new[idx].target.fitness, sense):
                    pop_new[idx] = agent
        self.objs = self.update_weight_agents__(pop_new)
        self.pop = self.mirror__()
