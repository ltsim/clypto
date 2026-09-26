#!/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
# --- dedicated agents (private to this module) ---

import numpy as np



from clypto.optimizer.native cimport utils as cy
from clypto.optimizer.native import ops
from clypto.optimizer.native.vectorize cimport VectorizeOptimizer
from clypto.optimizer.native.population cimport NativePopulation
from clypto.optimizer.native.agent cimport LegacyNativeAgent


class _SAPAgent:
    """A classic agent with its own crossover/mutation rates and population size."""

    __slots__ = ("solution", "target", "crossover", "mutation", "pop_size")

    def __init__(self, solution, target, crossover, mutation, pop_size):
        self.solution, self.target = solution, target
        self.crossover, self.mutation, self.pop_size = crossover, mutation, pop_size

    def copy(self):
        return _SAPAgent(self.solution, None if self.target is None else self.target.copy(),
                         self.crossover, self.mutation, self.pop_size)


cdef class SAP_DE(VectorizeOptimizer):
    """
    The original version of: Differential Evolution with Self-Adaptive Populations (SAP_DE)

    Links:
        1. https://doi.org/10.1007/s00500-005-0537-1

    Hyper-parameters should fine-tune in approximate range to get faster convergence toward the global optimum:
        + branch (str): ["ABS" or "REL"], gaussian (absolute) or uniform (relative) method

    Examples
    ~~~~~~~~
    >>> from clypto.native.collection.vectorize.evolutionary_based import DE    >>> import numpy as np
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
    >>> model = DE.SAP_DE(epoch=1000, pop_size=50, branch = "ABS")
    >>> g_best = model.solve(problem_dict)
    >>> print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
    >>> print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")

    References
    ~~~~~~~~~~
    [1] Teo, J., 2006. Exploring dynamic self-adaptive populations in differential evolution. Soft Computing, 10(8), pp.673-686.
    """

    cdef public object branch
    cdef public object fixed_pop_size
    cdef public object F
    cdef public object objs

    def __init__(
        self,
        epoch: int = 1000,
        pop_size: int = 100,
        branch: str = "ABS",
        *,
        name: str | None = None,
        mode: str | None = None,
    ) -> None:
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            branch (str): gaussian (absolute) or uniform (relative) method
        """
        VectorizeOptimizer.__init__(
            self,
            parameters=["epoch", "pop_size", "branch"],
            sort_flag=False,
            name=name,
            mode=mode,
        )
        self.epoch = cy.validator(int, epoch, [1, 100000], "epoch")
        self.pop_size = cy.validator(int, pop_size, [5, 10000], "pop_size")
        self.branch = cy.validator(str, branch, ["ABS", "REL"], "branch")
        self.fixed_pop_size = self.pop_size

    def new_agent__(self, solution=None, evaluate=False):
        """generate_empty_agent / generate_agent: the rates and the population size are drawn per agent."""
        if solution is None:
            solution = self.problem.generate_solution(True)
        crossover_rate = self.generator.uniform(0, 1)
        mutation_rate = self.generator.uniform(0, 1)
        if self.branch == "ABS":
            pop_size = int(10 * self.problem.n_dims + self.generator.normal(0, 1))
        else:  # elif self.branch == "REL":
            pop_size = int(10 * self.problem.n_dims + self.generator.uniform(-0.5, 0.5))
        agent = _SAPAgent(solution, None, crossover_rate, mutation_rate, pop_size)
        if evaluate:
            agent.target = self._get_target(agent.solution)
        return agent

    def mirror__(self):
        return ops.build_population(self, self.objs)

    def _initialization(self):
        # every agent carries its own rates and population size, so the agents are kept as objects
        if self._starting is not None:
            self.objs = [self.new_agent__(x, True) for x in self._starting]
        else:
            self.objs = [self.new_agent__(None, True) for _ in range(self.pop_size)]
        self.pop = self.mirror__()

    def edit_to_range__(self, var=None, lower=0, upper=1, func_value=None):
        while var <= lower or var >= upper:
            if var <= lower:
                var += func_value()
            if var >= upper:
                var -= func_value()
        return var

    def _evolve(self, int epoch_c):
        cdef bint swarm = self.mode in self.AVAILABLE_MODES
        popl = self.objs
        pop = []
        for idx in range(0, self.pop_size):
            # Choose 3 random element and different to idx
            idxs = self.generator.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
            j = self.generator.integers(0, self.pop_size)
            self.F = self.generator.normal(0, 1)
            ## Crossover
            if self.generator.uniform(0, 1) < popl[idx].crossover or idx == j:
                pos_new = popl[idxs[0]].solution + self.F * (popl[idxs[1]].solution - popl[idxs[2]].solution)
                cr_new = popl[idxs[0]].crossover + self.F * (popl[idxs[1]].crossover - popl[idxs[2]].crossover)
                mr_new = popl[idxs[0]].mutation + self.F * (popl[idxs[1]].mutation - popl[idxs[2]].mutation)
                if self.branch == "ABS":
                    ps_new = popl[idxs[0]].pop_size + int(self.F * (popl[idxs[1]].pop_size - popl[idxs[2]].pop_size))
                else:  # elif self.branch == "REL":
                    ps_new = popl[idxs[0]].pop_size + self.F * (popl[idxs[1]].pop_size - popl[idxs[2]].pop_size)
                pos_new = self._correct_solution(pos_new)
                cr_new = self.edit_to_range__(cr_new, 0, 1, self.generator.random)
                mr_new = self.edit_to_range__(mr_new, 0, 1, self.generator.random)
                agent = self.new_agent__(pos_new)
                pop.append(agent)
                if not swarm:
                    agent.target = self._get_target(pos_new)
                    agent.crossover, agent.mutation, agent.pop_size = cr_new, mr_new, ps_new
            else:
                pop.append(popl[idx].copy())
            ## Mutation
            if self.generator.uniform(0, 1) < popl[idxs[0]].mutation:
                pos_new = popl[idx].solution + self.generator.normal(0, popl[idxs[0]].mutation)
                cr_new = self.generator.normal(0, 1)
                mr_new = self.generator.normal(0, 1)
                if self.branch == "ABS":
                    ps_new = popl[idx].pop_size + int(self.generator.normal(0.5, 1))
                else:  # elif self.branch == "REL":
                    ps_new = popl[idx].pop_size + self.generator.normal(0, popl[idxs[0]].mutation)
                pos_new = self._correct_solution(pos_new)
                agent = self.new_agent__(pos_new)
                pop.append(agent)
                if not swarm:
                    agent.target = self._get_target(pos_new)
                    agent.crossover, agent.mutation, agent.pop_size = cr_new, mr_new, ps_new
        pop = ops.update_targets(self, pop)
        # Calculate new population size
        total = np.sum([pop[idx].pop_size for idx in range(0, self.pop_size)])
        if self.branch == "ABS":
            m_new = int(total / self.pop_size)
        else:  # elif self.branch == "REL":
            m_new = int(self.pop_size + total)
        if m_new <= 4:
            m_new = self.fixed_pop_size + int(self.generator.uniform(0, 4))
        elif m_new > 4 * self.fixed_pop_size:
            m_new = self.fixed_pop_size - int(self.generator.uniform(0, 4))
        ## Change population by population size
        if m_new <= self.pop_size:
            self.objs = pop[:m_new]
        else:
            pop_sorted = ops.sorted_agents(self, pop)
            self.objs = pop + pop_sorted[: m_new - self.pop_size]
        self.pop_size = len(self.objs)
        self.pop = self.mirror__()
